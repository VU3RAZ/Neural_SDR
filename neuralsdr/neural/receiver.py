"""
neural/receiver.py
==================
Unified Conditional Neural Receiver.

This is the central module of NeuralSDR — a single neural network that
handles all modulation modes through FiLM conditioning.

Full pipeline (end-to-end, 100% neural):
─────────────────────────────────────────
  Raw IQ [B, 2, T]
      │
      ▼
  [NeuralAGC]              ← learned gain control
      │
      ▼
  [NeuralChannelFilter]    ← learned bandpass, conditioned on bandwidth
      │
      ▼
  [NeuralFreqOffsetCorrector] ← learned carrier recovery
      │
      ▼
  [ConditioningEmbedding]  ← fuse modulation + frequency embeddings
      │
      ├───────────────────────────────────┐
      ▼                                   │ cond [B, 192]
  [ResNet1D Backbone]                     │ (FiLM applied at each block)
  4 stages, channels: 2→64→128→256→512   │
      │                                   │
      ▼                                   │
  [Transformer Encoder]    ◄──────────────┘
  6 layers, 8 heads, d=512, rotary PE
      │
      ├─────────────────┬─────────────────┐
      ▼                 ▼                 ▼
  [AnalogHead]   [DigitalHead]    [SquelchHead]
  PCM audio      Symbol logits    Signal presence
  [B,1,T_audio]  [B,N,M]          [B]

Architecture details
--------------------
ResNet1D
  - Each stage: 2× ResBlock with bottleneck (1×1 → 3×3 → 1×1 conv)
  - Stride-2 downsampling at each stage transition
  - FiLM conditioning applied after each block's BN
  - Activation: GELU throughout

Transformer Encoder
  - 6 layers of: LayerNorm → MHA → residual → LayerNorm → FFN → residual
  - Rotary Position Encoding (RoPE) — handles variable-length sequences
  - FiLM conditioning on each layer's output
  - FFN expansion factor: 4× with GELU

Parameters (default config)
  - ResNet stem + 4 stages: ~2.1M params
  - Transformer 6L×8H×512: ~18.9M params
  - Heads (analog + digital + squelch): ~3.2M params
  - Embeddings + conditioning: ~0.8M params
  Total: ~25M params — fits in 2 GB VRAM at float32
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .agc_norm       import NeuralAGC
from .channel_filter import NeuralChannelFilter
from .freq_offset    import NeuralFreqOffsetCorrector
from .embeddings     import ConditioningEmbedding, FiLMGenerator
from .heads          import AnalogHead, DigitalHead, SquelchHead
from ..config.settings import ModelConfig, ModulationMode


# ─────────────────────────────────────────────────────────────────────────────
# Rotary Position Encoding
# ─────────────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Encoding (RoPE) [Su et al., 2021].

    Applied to Q and K in each attention head.  Better than sinusoidal PE
    for sequences of varying length, and more sample-efficient.

    Parameters
    ----------
    dim      : int  — Head dimension (must be even).
    max_seq  : int  — Maximum sequence length.
    """

    def __init__(self, dim: int, max_seq: int = 4096) -> None:
        super().__init__()
        assert dim % 2 == 0
        theta = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq).float()
        freqs = torch.outer(positions, theta)                   # [max_seq, dim/2]
        cos = freqs.cos()
        sin = freqs.sin()
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary embedding to a sequence.

        Parameters
        ----------
        x : Tensor  [B, T, D]

        Returns
        -------
        Tensor  [B, T, D]
        """
        T, D = x.shape[1], x.shape[2]
        cos = self.cos[:T, :D // 2]   # [T, D/2]
        sin = self.sin[:T, :D // 2]

        # Split x into pairs
        x1, x2 = x[..., :D // 2], x[..., D // 2:]
        # Rotate
        x_rot = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ], dim=-1)
        return x_rot


# ─────────────────────────────────────────────────────────────────────────────
# ResNet1D building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    """
    1-D residual block with FiLM conditioning.

    Structure (pre-activation):
        x → BN → GELU → Conv1d(k=3) → BN → GELU → Conv1d(k=3) → FiLM → + x

    Parameters
    ----------
    channels  : int  — Number of channels.
    cond_dim  : int  — FiLM conditioning dimension.
    kernel    : int  — Convolution kernel size.
    """

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        kernel:   int = 7,
    ) -> None:
        super().__init__()
        pad = kernel // 2

        self.norm1 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad, bias=False)
        self.act   = nn.GELU()
        self.film  = FiLMGenerator(cond_dim=cond_dim, feature_dim=channels)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.film(h, cond)
        return x + self.scale * h


class ResStage1D(nn.Module):
    """
    One ResNet stage: optional downsampling + N residual blocks.

    Parameters
    ----------
    in_ch    : int  — Input channels.
    out_ch   : int  — Output channels.
    n_blocks : int  — Number of residual blocks.
    stride   : int  — Downsampling stride (1 = no downsampling).
    cond_dim : int  — FiLM conditioning dimension.
    kernel   : int  — Residual block kernel size.
    """

    def __init__(
        self,
        in_ch:    int,
        out_ch:   int,
        n_blocks: int = 2,
        stride:   int = 2,
        cond_dim: int = 192,
        kernel:   int = 7,
    ) -> None:
        super().__init__()

        # Downsampling projection
        if in_ch != out_ch or stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=stride, stride=stride, bias=False),
                nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            )
        else:
            self.downsample = nn.Identity()

        # Strided conv to change channels + downsample
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride,
                      padding=kernel // 2, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.GELU(),
        )

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResBlock1D(channels=out_ch, cond_dim=cond_dim, kernel=kernel)
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)                              # [B, out_ch, T/stride]
        for block in self.blocks:
            x = block(x, cond)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Encoder block
# ─────────────────────────────────────────────────────────────────────────────

class TransformerEncoderLayer(nn.Module):
    """
    Pre-norm Transformer encoder layer with RoPE and FiLM conditioning.

    Structure:
        x → LN → MHA(Q+RoPE, K+RoPE, V) → + x → LN → FFN → FiLM(cond) → + x

    Parameters
    ----------
    d_model  : int  — Model dimension.
    n_heads  : int  — Number of attention heads.
    ff_mult  : int  — FFN expansion factor.
    dropout  : float
    cond_dim : int  — FiLM conditioning dimension.
    rope     : RotaryEmbedding — Shared rotary embedding.
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        ff_mult:  int = 4,
        dropout:  float = 0.1,
        cond_dim: int = 192,
        rope:     Optional[RotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        head_dim = d_model // n_heads
        self.rope = rope

        # Self-attention
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # FFN
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

        # FiLM on FFN output
        self.film  = FiLMGenerator(cond_dim=cond_dim, feature_dim=d_model)

    def forward(
        self,
        x:    torch.Tensor,
        cond: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : Tensor  [B, T, D]
        cond : Tensor  [B, cond_dim]
        mask : Optional attention mask

        Returns
        -------
        Tensor  [B, T, D]
        """
        # ── Self-attention ────────────────────────────────────────────────────
        h = self.norm1(x)
        if self.rope is not None:
            # Apply RoPE to Q and K (not V)
            h = self.rope(h)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + attn_out

        # ── FFN with FiLM conditioning ────────────────────────────────────────
        h = self.norm2(x)
        h = self.ffn(h)
        h = self.film(h, cond)
        x = x + h

        return x


# ─────────────────────────────────────────────────────────────────────────────
# Unified Conditional Neural Receiver
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReceiverOutput:
    """
    All outputs from one forward pass of the neural receiver.

    Analog path  : audio_pcm (always computed for analog modes)
    Digital path : symbol_logits, bit_llr, timing_offset
    Common       : presence_prob, snr_db, freq_offset_norm, phase_offset_rad
    """
    # ── Analog ────────────────────────────────────────────────────────────────
    audio_pcm:        Optional[torch.Tensor]  = None   # [B, 1, T_audio]

    # ── Digital ───────────────────────────────────────────────────────────────
    symbol_logits:    Optional[torch.Tensor]  = None   # [B, N_syms, M]
    bit_llr:          Optional[torch.Tensor]  = None   # [B, N_syms, 6]
    timing_offset:    Optional[torch.Tensor]  = None   # [B, N_syms]

    # ── Squelch ───────────────────────────────────────────────────────────────
    presence_prob:    Optional[torch.Tensor]  = None   # [B]
    snr_db:           Optional[torch.Tensor]  = None   # [B]
    energy_profile:   Optional[torch.Tensor]  = None   # [B, T_enc]

    # ── Correction diagnostics ────────────────────────────────────────────────
    freq_offset_norm: Optional[torch.Tensor]  = None   # [B]
    phase_offset_rad: Optional[torch.Tensor]  = None   # [B]

    # ── Encoder features (for visualization / downstream) ─────────────────────
    encoder_output:   Optional[torch.Tensor]  = None   # [B, D, T_enc]


class NeuralReceiver(nn.Module):
    """
    Unified Conditional Neural Receiver.

    Single model that handles all supported modulation modes via FiLM
    conditioning.  The modulation mode is passed as an integer index at
    inference time, enabling real-time mode switching without reloading weights.

    Parameters
    ----------
    cfg : ModelConfig — Architecture hyperparameters.

    Usage
    -----
    receiver = NeuralReceiver(cfg)

    # Prepare batch
    iq_tensor  = iq_to_tensor(iq_chunk).unsqueeze(0)   # [1, 2, T]
    mod_idx    = torch.tensor([ModulationMode.FM_WB.index])
    center_f   = torch.tensor([100.0e6])
    bandwidth  = torch.tensor([200e3])

    with torch.no_grad():
        out = receiver(iq_tensor, mod_idx, center_f, bandwidth)
    audio_np = out.audio_pcm.squeeze().numpy()
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        cond_dim = cfg.mod_embed_dim + cfg.freq_embed_dim

        # ── Pre-processing blocks ─────────────────────────────────────────────
        self.agc    = NeuralAGC(
            agc_window=cfg.agc_window,
            attack_alpha=cfg.agc_attack,
            release_alpha=cfg.agc_release,
        )
        self.ch_filter = NeuralChannelFilter(
            n_layers=4,
            kernel_size=cfg.filter_taps // 4,
            n_channels=2,
            max_bandwidth_hz=cfg.model_fields["max_freq_offset_norm"].default
                             if False else 500_000.0,
        )
        self.freq_corr = NeuralFreqOffsetCorrector(
            max_offset_norm=cfg.max_freq_offset_norm,
        )

        # ── Conditioning embedding ────────────────────────────────────────────
        self.conditioning = ConditioningEmbedding(
            mod_embed_dim=cfg.mod_embed_dim,
            freq_embed_dim=cfg.freq_embed_dim,
            out_dim=cond_dim,
        )

        # ── ResNet1D backbone ─────────────────────────────────────────────────
        channels = cfg.resnet_channels   # e.g. [64, 128, 256, 512]
        self.resnet_stem = nn.Sequential(
            nn.Conv1d(2, channels[0], kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(num_groups=min(8, channels[0]), num_channels=channels[0]),
            nn.GELU(),
        )
        self.resnet_stages = nn.ModuleList([
            ResStage1D(
                in_ch=channels[i],
                out_ch=channels[i + 1] if i + 1 < len(channels) else channels[-1],
                n_blocks=2,
                stride=2,
                cond_dim=cond_dim,
                kernel=cfg.resnet_kernel,
            )
            for i in range(len(channels) - 1)
        ])
        d_model = channels[-1]   # Transformer dimension = final ResNet channels

        # Projection to transformer dimension if needed
        if d_model != cfg.transformer_dim:
            self.res_to_trans = nn.Conv1d(d_model, cfg.transformer_dim, kernel_size=1)
            d_model = cfg.transformer_dim
        else:
            self.res_to_trans = nn.Identity()

        # ── Transformer encoder ───────────────────────────────────────────────
        self.rope = RotaryEmbedding(dim=d_model // cfg.transformer_heads)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                n_heads=cfg.transformer_heads,
                ff_mult=cfg.transformer_ff_mult,
                dropout=cfg.dropout,
                cond_dim=cond_dim,
                rope=self.rope,
            )
            for _ in range(cfg.transformer_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # ── Output heads ─────────────────────────────────────────────────────
        self.analog_head = AnalogHead(
            encoder_dim=d_model,
            upsample_factor=cfg.analog_upsample_factor,
            cond_dim=cond_dim,
        )
        self.digital_head = DigitalHead(
            encoder_dim=d_model,
            samples_per_sym=8,    # approximate, refined by timing head
            cond_dim=cond_dim,
        )
        self.squelch_head = SquelchHead(
            encoder_dim=d_model,
            cond_dim=cond_dim,
        )

        # Store model dimension for downstream use
        self.d_model = d_model

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply sensible weight initialization across the model."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        iq:             torch.Tensor,
        mod_indices:    torch.Tensor,
        center_freq_hz: torch.Tensor,
        bandwidth_hz:   Optional[torch.Tensor]  = None,
        mod_mode:       Optional[ModulationMode] = None,
        run_analog:     bool = True,
        run_digital:    bool = True,
    ) -> ReceiverOutput:
        """
        Full forward pass.

        Parameters
        ----------
        iq             : Tensor  [B, 2, T]  — raw IQ (I=ch0, Q=ch1)
        mod_indices    : Tensor  [B]        — modulation mode indices
        center_freq_hz : Tensor  [B]        — centre frequencies (Hz)
        bandwidth_hz   : Tensor  [B] or None — signal bandwidth (Hz)
        mod_mode       : ModulationMode or None — for digital head masking
        run_analog     : bool — whether to run the analog head
        run_digital    : bool — whether to run the digital head

        Returns
        -------
        ReceiverOutput
        """
        B = iq.shape[0]

        # ── Stage 1: Pre-processing ───────────────────────────────────────────
        x = self.agc(iq)                                     # [B, 2, T]
        x = self.ch_filter(x, bandwidth_hz)                  # [B, 2, T]
        x, freq_off, phase_off = self.freq_corr(x)          # [B, 2, T], [B], [B]

        # ── Stage 2: Conditioning embedding ──────────────────────────────────
        cond = self.conditioning(mod_indices, center_freq_hz)  # [B, cond_dim]

        # ── Stage 3: ResNet1D backbone ────────────────────────────────────────
        h = self.resnet_stem(x)                              # [B, C0, T]
        for stage in self.resnet_stages:
            h = stage(h, cond)                               # [B, Ci, T/2^i]

        h = self.res_to_trans(h)                             # [B, d_model, T_enc]

        # ── Stage 4: Transformer encoder ─────────────────────────────────────
        # Convert to sequence format [B, T_enc, D]
        h_seq = h.transpose(1, 2)
        for layer in self.transformer_layers:
            h_seq = layer(h_seq, cond)
        h_seq = self.encoder_norm(h_seq)                     # [B, T_enc, D]

        # Back to conv format [B, D, T_enc]
        enc_out = h_seq.transpose(1, 2)

        # ── Stage 5: Squelch (always run) ────────────────────────────────────
        squelch_result = self.squelch_head(enc_out, cond)

        # ── Stage 6: Task heads (conditional) ────────────────────────────────
        audio_pcm     = None
        symbol_logits = None
        bit_llr       = None
        timing_offset = None

        if run_analog:
            audio_pcm = self.analog_head(enc_out, cond)      # [B, 1, T_audio]

        if run_digital:
            dig_out = self.digital_head(enc_out, cond, mod_mode)
            symbol_logits = dig_out["symbol_logits"]
            bit_llr       = dig_out["bit_llr"]
            timing_offset = dig_out["timing_offset"]

        return ReceiverOutput(
            audio_pcm        = audio_pcm,
            symbol_logits    = symbol_logits,
            bit_llr          = bit_llr,
            timing_offset    = timing_offset,
            presence_prob    = squelch_result["presence_prob"],
            snr_db           = squelch_result["snr_db"],
            energy_profile   = squelch_result["energy_profile"],
            freq_offset_norm = freq_off,
            phase_offset_rad = phase_off,
            encoder_output   = enc_out,
        )

    @torch.no_grad()
    def infer(
        self,
        iq_numpy,              # np.ndarray complex64 [T]
        mod_mode: ModulationMode,
        center_freq_hz: float,
        bandwidth_hz:   float,
        device: torch.device,
    ) -> ReceiverOutput:
        """
        Convenience method for single-chunk inference from numpy IQ.

        Handles tensor preparation, device placement, and mode selection.
        """
        from ..dsp.iq_utils import iq_to_tensor

        # Prepare input
        iq_t = iq_to_tensor(iq_numpy, device=device).unsqueeze(0)   # [1, 2, T]
        mod_idx = torch.tensor([mod_mode.index], device=device)
        cf  = torch.tensor([center_freq_hz], device=device)
        bw  = torch.tensor([bandwidth_hz], device=device)

        return self.forward(
            iq_t, mod_idx, cf, bw,
            mod_mode   = mod_mode,
            run_analog = mod_mode.is_analog,
            run_digital= mod_mode.is_digital,
        )

    def count_parameters(self) -> Dict[str, int]:
        """Return parameter count broken down by sub-module."""
        def count(m): return sum(p.numel() for p in m.parameters())
        return {
            "agc":             count(self.agc),
            "channel_filter":  count(self.ch_filter),
            "freq_corrector":  count(self.freq_corr),
            "conditioning":    count(self.conditioning),
            "resnet_backbone": count(self.resnet_stem) + sum(count(s) for s in self.resnet_stages),
            "transformer":     sum(count(l) for l in self.transformer_layers),
            "analog_head":     count(self.analog_head),
            "digital_head":    count(self.digital_head),
            "squelch_head":    count(self.squelch_head),
            "total":           count(self),
        }

    def print_summary(self) -> None:
        """Print a concise parameter count summary."""
        counts = self.count_parameters()
        print("\n" + "=" * 52)
        print("  NeuralReceiver — Parameter Summary")
        print("=" * 52)
        for name, n in counts.items():
            if name == "total":
                print("-" * 52)
            tag = " (total)" if name == "total" else ""
            print(f"  {name:<22} : {n:>10,}{tag}")
        print("=" * 52 + "\n")
