"""
neural/heads/analog_head.py
===========================
Analog waveform reconstruction head.

Takes the encoder's latent representation and reconstructs PCM audio for
analog modulations (AM, FM, SSB, CW, DSB).

Architecture
------------
  Input : [B, D, T_enc]    (encoder output, T_enc = T // stride_product)
  Output: [B, 1, T_audio]  (mono PCM audio at audio sample rate)

  Transposed convolution upsampling stack:
  ┌─────────────────────────────────────────────────┐
  │  [B, D, T_enc]                                  │
  │  → FiLM(cond)                                   │
  │  → TransposeConv(D→D//2, k=4, s=2) → BN → GELU │  ×upsample_stages
  │  → TransposeConv(D//2→1, k=4, s=2) → Tanh       │
  │  → [B, 1, T_audio]                              │
  └─────────────────────────────────────────────────┘

Why transposed convolution?
- Avoids interpolation artifacts (vs. nearest/bilinear + conv).
- Fully learnable upsampling kernel.
- Has been validated in WaveNet/WaveGlow-style architectures.

The Tanh output keeps audio in [-1, 1] range (normalized float PCM).
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..embeddings import FiLMGenerator


class UpsampleBlock(nn.Module):
    """
    One upsampling stage: TransposedConv1d → GroupNorm → GELU.

    Parameters
    ----------
    in_channels  : int
    out_channels : int
    upsample_factor : int — Stride of the transposed conv (= upsampling ratio).
    kernel_size  : int — Kernel size (should be ≥ upsample_factor, even multiple).
    cond_dim     : int — Conditioning vector dimension for FiLM.
    """

    def __init__(
        self,
        in_channels:     int,
        out_channels:    int,
        upsample_factor: int = 2,
        kernel_size:     int = 4,
        cond_dim:        int = 192,
    ) -> None:
        super().__init__()
        self.upsample_factor = upsample_factor

        # Trim kernel to avoid checkerboard: kernel = stride * N (even multiple)
        assert kernel_size >= upsample_factor
        self.conv_t = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=upsample_factor,
            padding=(kernel_size - upsample_factor) // 2,
        )
        self.norm = nn.GroupNorm(
            num_groups=min(8, out_channels),
            num_channels=out_channels,
        )
        self.act  = nn.GELU()
        self.film = FiLMGenerator(cond_dim=cond_dim, feature_dim=out_channels)

        # Residual path (1×1 transposed conv for shape matching)
        if in_channels != out_channels:
            self.res_conv = nn.ConvTranspose1d(
                in_channels, out_channels,
                kernel_size=upsample_factor, stride=upsample_factor,
                padding=0,
            )
        else:
            self.res_conv = nn.ConvTranspose1d(
                in_channels, out_channels,
                kernel_size=upsample_factor, stride=upsample_factor,
                padding=0,
            )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : Tensor  [B, C_in, T]
        cond : Tensor  [B, cond_dim]

        Returns
        -------
        Tensor  [B, C_out, T * upsample_factor]
        """
        residual = self.res_conv(x)
        h = self.conv_t(x)
        # Trim any length mismatch from the transposed conv
        if h.shape[-1] != residual.shape[-1]:
            h = h[:, :, :residual.shape[-1]]
        h = self.norm(h)
        h = self.film(h, cond)
        h = self.act(h)
        return h + residual


class AnalogHead(nn.Module):
    """
    Analog waveform reconstruction head.

    Progressively upsamples the encoder's time-compressed representation
    back to audio sample rate, producing normalized float PCM audio.

    Parameters
    ----------
    encoder_dim      : int   — Encoder output channel dimension.
    upsample_factor  : int   — Total upsampling ratio (e.g. 4 = 2 stages of 2×).
    cond_dim         : int   — Conditioning embedding dimension.
    output_sr        : int   — Target audio sample rate (for metadata; not used
                               in computation — the model learns the mapping).
    """

    def __init__(
        self,
        encoder_dim:    int = 512,
        upsample_factor: int = 4,
        cond_dim:       int = 192,
        output_sr:      int = 44100,
    ) -> None:
        super().__init__()
        self.encoder_dim    = encoder_dim
        self.upsample_factor = upsample_factor
        self.output_sr      = output_sr

        # Decompose total upsample into stages of 2×
        n_stages = int(math.log2(upsample_factor))
        assert 2 ** n_stages == upsample_factor, \
            f"upsample_factor must be a power of 2, got {upsample_factor}"

        # Build upsample stack: D → D//2 → D//4 → ... → 16
        channels = [max(16, encoder_dim // (2 ** i)) for i in range(n_stages + 1)]
        channels[-1] = 16   # bottleneck before final projection

        self.upsample_blocks = nn.ModuleList([
            UpsampleBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                upsample_factor=2,
                kernel_size=4,
                cond_dim=cond_dim,
            )
            for i in range(n_stages)
        ])

        # Final projection to mono audio with Tanh
        self.final_proj = nn.Sequential(
            nn.Conv1d(channels[-1], 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=1),
            nn.Tanh(),
        )

        # Input projection (handles encoder_dim ≠ channels[0])
        if encoder_dim != channels[0]:
            self.in_proj = nn.Conv1d(encoder_dim, channels[0], kernel_size=1)
        else:
            self.in_proj = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : Tensor  [B, encoder_dim, T_enc]  — encoder latent
        cond : Tensor  [B, cond_dim]            — conditioning vector

        Returns
        -------
        audio : Tensor  [B, 1, T_audio]  — PCM audio, range [-1, 1]
                where T_audio = T_enc × upsample_factor
        """
        h = self.in_proj(x)                        # [B, channels[0], T_enc]

        for block in self.upsample_blocks:
            h = block(h, cond)                     # 2× upsampling each time

        audio = self.final_proj(h)                 # [B, 1, T_audio]
        return audio

    def output_length(self, encoder_length: int) -> int:
        """Compute output audio length given encoder sequence length."""
        return encoder_length * self.upsample_factor
