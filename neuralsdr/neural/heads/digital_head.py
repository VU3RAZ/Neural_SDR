"""
neural/heads/digital_head.py
============================
Digital symbol and bit output head.

Takes the encoder's latent and produces soft symbol probabilities for
digital modulations (BPSK, QPSK, 8PSK, 16QAM, 64QAM, GFSK, CPFSK, OFDM).

Architecture
------------
  Input : [B, D, T_enc]      (encoder output)
  Output:
    symbol_logits  [B, N_syms, M]   — log-softmax over M constellation pts
    bit_logits     [B, N_bits]      — per-bit LLR (for Viterbi / FEC)
    timing_offset  [B, N_syms]      — fractional symbol timing estimates

  Structure
  ---------
  [B, D, T_enc]
  → FiLM(cond)
  → Symbol extractor: Conv1d(D→D, k=sps, s=sps//2) — coarse timing
  → Temporal attention over symbols
  → Per-symbol MLP head → symbol_logits [B, N_syms, M]
  → Bit deinterleaver → bit_logits [B, N_bits]
  → Timing head → timing_offset [B, N_syms]

Design notes
------------
- The symbol extractor uses a strided conv to move from sample-rate to
  approximately symbol-rate, with overlap for timing recovery.
- The max constellation size is 64 (for 64-QAM).  For lower-order
  modulations, only the first M logits are used (the rest are masked).
- Soft outputs (log-softmax) allow computing bit LLRs via marginalisation,
  which is needed for soft-decision FEC.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..embeddings import FiLMGenerator
from ...config.settings import ModulationMode


# Maximum constellation size supported
MAX_CONSTELLATION = 64


class TimingHead(nn.Module):
    """
    Estimates fractional symbol timing offsets.

    Outputs a value in [-0.5, 0.5] for each symbol window, representing
    the fractional sample offset of the optimal sampling point.

    Parameters
    ----------
    in_dim : int — Per-symbol feature dimension.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Tanh(),                   # output in [-1, 1], scaled to [-0.5, 0.5]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  [B, N_syms, D]

        Returns
        -------
        Tensor  [B, N_syms]  — timing offsets in (-0.5, 0.5)
        """
        return self.net(x).squeeze(-1) * 0.5


class SymbolAttention(nn.Module):
    """
    Lightweight self-attention over symbol windows.

    Captures inter-symbol dependencies (e.g., Nyquist ISI, OFDM guard intervals).

    Parameters
    ----------
    dim     : int — Feature dimension.
    n_heads : int — Number of attention heads.
    """

    def __init__(self, dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  [B, N_syms, D]

        Returns
        -------
        Tensor  [B, N_syms, D]
        """
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


class DigitalHead(nn.Module):
    """
    Multi-modulation digital symbol output head.

    Handles all digital modulations through a shared architecture with
    modulation-conditioned output masking.

    Parameters
    ----------
    encoder_dim     : int  — Encoder output channels.
    samples_per_sym : int  — Approximate samples per symbol at input rate.
    cond_dim        : int  — Conditioning vector dimension.
    n_attn_heads    : int  — Heads in symbol-level attention.
    """

    def __init__(
        self,
        encoder_dim:     int = 512,
        samples_per_sym: int = 8,
        cond_dim:        int = 192,
        n_attn_heads:    int = 4,
    ) -> None:
        super().__init__()
        self.encoder_dim     = encoder_dim
        self.samples_per_sym = samples_per_sym
        sym_dim = 256          # per-symbol feature dimension

        # ── FiLM conditioning on encoder features ─────────────────────────────
        self.input_film = FiLMGenerator(cond_dim=cond_dim, feature_dim=encoder_dim)

        # ── Symbol extractor: sample-rate → symbol-rate ───────────────────────
        # Strided conv with overlap (stride = sps // 2 for coarse timing recovery)
        stride = max(1, samples_per_sym // 2)
        self.symbol_extractor = nn.Sequential(
            nn.Conv1d(
                encoder_dim, sym_dim,
                kernel_size=samples_per_sym,
                stride=stride,
                padding=samples_per_sym // 2,
                bias=False,
            ),
            nn.GroupNorm(num_groups=min(16, sym_dim), num_channels=sym_dim),
            nn.GELU(),
            nn.Conv1d(sym_dim, sym_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # ── Per-symbol attention ──────────────────────────────────────────────
        self.sym_attn = SymbolAttention(dim=sym_dim, n_heads=n_attn_heads)

        # ── FiLM on symbol features ───────────────────────────────────────────
        self.sym_film = FiLMGenerator(cond_dim=cond_dim, feature_dim=sym_dim)

        # ── Symbol classifier: sym_dim → MAX_CONSTELLATION logits ─────────────
        self.symbol_head = nn.Sequential(
            nn.Linear(sym_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, MAX_CONSTELLATION),
        )

        # ── Bit LLR head: sym_dim → bits per symbol (log2(MAX_CONSTELLATION)=6) ─
        self.bit_head = nn.Sequential(
            nn.Linear(sym_dim, 64),
            nn.GELU(),
            nn.Linear(64, int(math.log2(MAX_CONSTELLATION))),   # 6 bits max
        )

        # ── Timing estimation ─────────────────────────────────────────────────
        self.timing_head = TimingHead(in_dim=sym_dim)

    def forward(
        self,
        x:          torch.Tensor,
        cond:       torch.Tensor,
        mod_mode:   Optional[ModulationMode] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x        : Tensor  [B, D, T_enc]  — encoder latent
        cond     : Tensor  [B, cond_dim]  — conditioning vector
        mod_mode : ModulationMode or None — current demod mode (for masking)

        Returns
        -------
        Dict with keys:
            'symbol_logits'  : Tensor [B, N_syms, M]  — per-symbol log-probs
            'bit_llr'        : Tensor [B, N_syms, 6]  — per-bit LLRs
            'timing_offset'  : Tensor [B, N_syms]     — timing offsets (frac. sample)
            'symbol_features': Tensor [B, N_syms, sym_dim]  — for downstream use
        """
        # ── FiLM-condition the encoder features ──────────────────────────────
        x = self.input_film(x, cond)                      # [B, D, T_enc]

        # ── Extract per-symbol features ───────────────────────────────────────
        sym_features = self.symbol_extractor(x)           # [B, sym_dim, N_syms]
        sym_features = sym_features.transpose(1, 2)       # [B, N_syms, sym_dim]

        # ── Symbol-level attention ────────────────────────────────────────────
        sym_features = self.sym_attn(sym_features)        # [B, N_syms, sym_dim]

        # ── FiLM-condition symbol features ────────────────────────────────────
        sym_features = self.sym_film(sym_features, cond)  # [B, N_syms, sym_dim]

        # ── Symbol logits ─────────────────────────────────────────────────────
        raw_logits = self.symbol_head(sym_features)       # [B, N_syms, MAX_CONS]

        # Mask logits for the active modulation
        if mod_mode is not None:
            M = mod_mode.num_constellation_points
            mask = torch.full_like(raw_logits, float('-inf'))
            mask[:, :, :M] = raw_logits[:, :, :M]
            symbol_logits = F.log_softmax(mask, dim=-1)
        else:
            symbol_logits = F.log_softmax(raw_logits, dim=-1)

        # ── Bit LLRs ──────────────────────────────────────────────────────────
        bit_llr = self.bit_head(sym_features)             # [B, N_syms, 6]

        # ── Timing offsets ────────────────────────────────────────────────────
        timing_offset = self.timing_head(sym_features)    # [B, N_syms]

        return {
            "symbol_logits":   symbol_logits,
            "bit_llr":         bit_llr,
            "timing_offset":   timing_offset,
            "symbol_features": sym_features,
        }

    def decode_bits(
        self,
        symbol_logits: torch.Tensor,
        mod_mode: ModulationMode,
    ) -> torch.Tensor:
        """
        Hard-decision bit decoding from symbol log-probs (Gray-coded).

        Parameters
        ----------
        symbol_logits : Tensor  [B, N_syms, M]  — log-softmax outputs
        mod_mode      : ModulationMode

        Returns
        -------
        Tensor  [B, N_syms * bits_per_sym]  — decoded bits (0/1)
        """
        M = mod_mode.num_constellation_points
        bits_per_sym = int(math.log2(M))
        symbols = symbol_logits[:, :, :M].argmax(dim=-1)  # [B, N_syms]

        # Convert symbol indices to bits via Gray decoding
        # Gray decode: gray ^ (gray >> 1) ^ (gray >> 2) ... (iterative)
        bits = []
        gray = symbols
        for shift in range(bits_per_sym - 1, -1, -1):
            bits.append((gray >> shift) & 1)
        bit_tensor = torch.stack(bits, dim=-1)           # [B, N_syms, bps]
        return bit_tensor.reshape(symbol_logits.shape[0], -1).float()
