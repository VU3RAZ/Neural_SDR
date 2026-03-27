"""
neural/channel_filter.py
========================
Neural channel selection and filtering block.

Replaces classical FIR/IIR bandpass filters with a stack of learned
depthwise-separable convolutions conditioned on the desired bandwidth.

Architecture
------------
  Input  : [B, 2, T]  (normalized IQ from AGC)
  Cond   : [B, 1]     (bandwidth, log-normalized to [0, 1])
  Output : [B, 2, T]  (channel-selected IQ)

  Layer stack
  ───────────
  ┌─ DepthwiseFilterBlock (×N_LAYERS) ─────────────────────────────┐
  │  Conv1d (depthwise, K taps) → BN → GELU → FiLM(bandwidth)     │
  │  + residual                                                     │
  └─────────────────────────────────────────────────────────────────┘
  → Pointwise 2→2 projection (learned I/Q mixing)
  → Output

Design rationale
----------------
- Depthwise convolutions are equivalent to independent FIR filters on I and Q.
- The kernel length controls the effective frequency selectivity.
- FiLM conditioning on bandwidth allows one model to handle everything from
  a 500 Hz CW filter to a 200 kHz FM broadcast filter.
- Stacking multiple short depthwise layers (vs. one very long layer) gives
  a receptive field equal to their sum while using far fewer parameters.
- Residual connections prevent gradient vanishing in deep filter stacks.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# FiLM conditioning layer (bandwidth)
# ─────────────────────────────────────────────────────────────────────────────

class BandwidthFiLM(nn.Module):
    """
    Feature-wise Linear Modulation conditioned on a scalar bandwidth value.

    Given a bandwidth scalar b ∈ [0, 1] (log-normalised), produces
    per-channel γ (scale) and β (bias) vectors that modulate feature maps.

    Parameters
    ----------
    n_channels : int — Number of feature map channels to modulate.
    hidden_dim : int — Hidden size of the FiLM generator MLP.
    """

    def __init__(self, n_channels: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_channels * 2),  # γ and β
        )
        # Initialize to identity: γ=1, β=0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias[:n_channels], 1.0)   # γ → 1
        nn.init.zeros_(self.net[-1].bias[n_channels:])            # β → 0

    def forward(
        self,
        x: torch.Tensor,
        bw_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x       : Tensor  [B, C, T]
        bw_norm : Tensor  [B, 1]   — normalized bandwidth in [0, 1]

        Returns
        -------
        Tensor  [B, C, T]
        """
        params = self.net(bw_norm)                   # [B, 2*C]
        C = x.shape[1]
        gamma = params[:, :C].unsqueeze(-1)          # [B, C, 1]
        beta  = params[:, C:].unsqueeze(-1)          # [B, C, 1]
        return gamma * x + beta


# ─────────────────────────────────────────────────────────────────────────────
# Single depthwise filter block
# ─────────────────────────────────────────────────────────────────────────────

class DepthwiseFilterBlock(nn.Module):
    """
    One building block of the neural channel filter.

    Structure (pre-activation residual):
        x → GroupNorm → GELU → DepthwiseConv1d → FiLM(bw) → + x

    The depthwise conv acts as a learned FIR filter on each channel
    independently (I channel filtered separately from Q channel).

    Parameters
    ----------
    channels   : int  — Number of channels (2 for raw IQ, more after projection).
    kernel_size: int  — Filter tap count (controls frequency resolution).
    groups     : int  — Depthwise groups (must divide channels evenly).
    bw_hidden  : int  — Hidden dim for FiLM generator.
    dilation   : int  — Dilation for increased receptive field without more params.
    """

    def __init__(
        self,
        channels:    int = 2,
        kernel_size: int = 64,
        groups:      int = 2,
        bw_hidden:   int = 32,
        dilation:    int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Pre-norm
        self.norm = nn.GroupNorm(num_groups=min(2, channels), num_channels=channels)

        # Depthwise causal convolution
        effective_kernel = (kernel_size - 1) * dilation + 1
        self.causal_pad = effective_kernel - 1
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        # Activation
        self.act = nn.GELU()

        # FiLM bandwidth conditioning
        self.film = BandwidthFiLM(n_channels=channels, hidden_dim=bw_hidden)

        # Learnable residual scale (initialized near zero for stable training)
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)

        # Sinc-initialized weights for better initial filter behaviour
        self._sinc_init()

    def _sinc_init(self) -> None:
        """Initialize conv weights with a sinc-like impulse (identity filter)."""
        with torch.no_grad():
            K = self.kernel_size
            center = K // 2
            w = torch.zeros_like(self.conv.weight)
            # Set center tap to 1.0 (identity / allpass initialization)
            w[:, :, center] = 1.0 / self.conv.groups
            self.conv.weight.copy_(w)

    def forward(
        self,
        x: torch.Tensor,
        bw_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x       : Tensor  [B, C, T]
        bw_norm : Tensor  [B, 1]

        Returns
        -------
        Tensor  [B, C, T]
        """
        residual = x
        h = self.norm(x)
        h = self.act(h)

        # Causal padding: pad left only so output doesn't look into the future
        h = F.pad(h, (self.causal_pad, 0))
        h = self.conv(h)                     # [B, C, T]

        # Bandwidth-conditioned FiLM modulation
        h = self.film(h, bw_norm)

        return residual + self.res_scale * h


# ─────────────────────────────────────────────────────────────────────────────
# Full neural channel filter
# ─────────────────────────────────────────────────────────────────────────────

class NeuralChannelFilter(nn.Module):
    """
    Multi-layer neural channel selection and filtering block.

    Stacks ``n_layers`` DepthwiseFilterBlocks with increasing dilation,
    providing an exponentially growing receptive field similar to WaveNet.
    FiLM conditioning on the normalised bandwidth allows the filter to
    adapt its passband without any weight updates.

    Parameters
    ----------
    n_layers        : int   — Number of filter blocks.
    kernel_size     : int   — Taps per block.
    n_channels      : int   — Internal channel width (2 for raw IQ; may be
                              expanded if injecting into a wider backbone).
    bw_hidden       : int   — FiLM hidden dimension.
    max_bandwidth_hz: float — Maximum bandwidth used for log-normalization.
    min_bandwidth_hz: float — Minimum bandwidth used for log-normalization.
    """

    def __init__(
        self,
        n_layers:         int   = 4,
        kernel_size:      int   = 32,
        n_channels:       int   = 2,
        bw_hidden:        int   = 32,
        max_bandwidth_hz: float = 500_000.0,
        min_bandwidth_hz: float = 100.0,
    ) -> None:
        super().__init__()

        self.min_bw_log = math.log(min_bandwidth_hz)
        self.max_bw_log = math.log(max_bandwidth_hz)

        # Build filter stack with exponentially increasing dilation
        self.blocks = nn.ModuleList([
            DepthwiseFilterBlock(
                channels=n_channels,
                kernel_size=kernel_size,
                groups=n_channels,   # fully depthwise (one filter per channel)
                bw_hidden=bw_hidden,
                dilation=2 ** i,
            )
            for i in range(n_layers)
        ])

        # Final pointwise mixer: allows cross-channel (I↔Q) information flow
        self.iq_mix = nn.Conv1d(n_channels, n_channels, kernel_size=1, bias=True)
        nn.init.eye_(self.iq_mix.weight.view(n_channels, n_channels))
        nn.init.zeros_(self.iq_mix.bias)

        # Output norm
        self.out_norm = nn.GroupNorm(num_groups=min(2, n_channels), num_channels=n_channels)

    def _normalize_bandwidth(
        self,
        bandwidth_hz: torch.Tensor,
    ) -> torch.Tensor:
        """
        Log-normalize bandwidth to [0, 1] for FiLM conditioning.

        Parameters
        ----------
        bandwidth_hz : Tensor  [B] or scalar

        Returns
        -------
        Tensor  [B, 1]
        """
        bw = bandwidth_hz.float().clamp(
            min=math.exp(self.min_bw_log),
            max=math.exp(self.max_bw_log),
        )
        bw_log = torch.log(bw)
        bw_norm = (bw_log - self.min_bw_log) / (self.max_bw_log - self.min_bw_log)
        return bw_norm.view(-1, 1)

    def forward(
        self,
        x: torch.Tensor,
        bandwidth_hz: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x            : Tensor  [B, 2, T]  — normalized IQ from AGC
        bandwidth_hz : Tensor  [B] or scalar  — desired signal bandwidth in Hz
                       None → use default wide-open filter (no conditioning)

        Returns
        -------
        Tensor  [B, 2, T]  — channel-filtered IQ
        """
        B = x.shape[0]

        if bandwidth_hz is None:
            # Default: normalize to mid-range bandwidth
            bw_norm = torch.full((B, 1), 0.5, device=x.device, dtype=x.dtype)
        else:
            if not isinstance(bandwidth_hz, torch.Tensor):
                bandwidth_hz = torch.tensor(
                    [bandwidth_hz] * B, device=x.device, dtype=x.dtype
                )
            bw_norm = self._normalize_bandwidth(bandwidth_hz)

        # Apply filter stack
        for block in self.blocks:
            x = block(x, bw_norm)

        # Cross-channel mixer
        x = self.iq_mix(x)
        x = self.out_norm(x)

        return x

    @property
    def receptive_field(self) -> int:
        """Effective receptive field in samples (sum of dilated kernel spans)."""
        return sum(
            (b.kernel_size - 1) * b.dilation + 1
            for b in self.blocks
        )
