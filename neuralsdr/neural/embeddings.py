"""
neural/embeddings.py
====================
Modulation and frequency conditioning embeddings for the Unified Neural Receiver.

This module provides the "brain" of the conditioning system that turns one
general-purpose neural backbone into a modulation-aware receiver.

Two embedding types
-------------------
1. ModulationEmbedding
   - Learned lookup table: modulation index → dense vector of dim mod_embed_dim
   - One entry per ModulationMode (15 entries initially, extensible)
   - Used as both a direct feature injection AND as a FiLM conditioning signal

2. FrequencyEmbedding
   - Log-scaled sinusoidal encoding of the centre frequency
   - Allows the model to use frequency priors (e.g., "FM broadcast is always
     around 88–108 MHz and has 75 kHz deviation")

FiLM Conditioning
-----------------
Feature-wise Linear Modulation (FiLM) [Perez et al., 2018]:
    FiLM(x; γ, β) = γ ⊙ x + β
where γ and β are produced by a small MLP from the conditioning embedding.

This is applied after every ResNet block and Transformer layer in the backbone,
giving the model per-layer, per-channel control over its representations based
on the active modulation mode.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.settings import ModulationMode, NUM_MODULATIONS


# ─────────────────────────────────────────────────────────────────────────────
# Modulation embedding
# ─────────────────────────────────────────────────────────────────────────────

class ModulationEmbedding(nn.Module):
    """
    Learned modulation embedding table.

    Maps a modulation mode index (integer) to a dense vector.
    The embedding is shared across all FiLM layers — each FiLM generator
    projects from this same embedding to its specific γ/β dimensions.

    Parameters
    ----------
    num_mods  : int — Number of modulation modes (default: len(ModulationMode)).
    embed_dim : int — Embedding dimension.
    dropout   : float — Embedding dropout for regularisation.
    """

    def __init__(
        self,
        num_mods:  int   = NUM_MODULATIONS,
        embed_dim: int   = 128,
        dropout:   float = 0.05,
    ) -> None:
        super().__init__()
        self.num_mods  = num_mods
        self.embed_dim = embed_dim

        self.embed   = nn.Embedding(num_mods, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(embed_dim)

        # Init: small normal — modulations start as similar, then diverge
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, mod_indices: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        mod_indices : Tensor  [B]  — Integer modulation indices (0..NUM_MODS-1)

        Returns
        -------
        Tensor  [B, embed_dim]
        """
        x = self.embed(mod_indices.long())   # [B, embed_dim]
        x = self.norm(x)
        return self.dropout(x)

    @classmethod
    def index_for(cls, mode: ModulationMode) -> int:
        """Return the embedding index for a ModulationMode enum value."""
        return mode.index

    def embed_mode(
        self,
        mode: ModulationMode,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Convenience: embed a single mode for a full batch.

        Parameters
        ----------
        mode       : ModulationMode
        batch_size : int
        device     : torch.device

        Returns
        -------
        Tensor  [B, embed_dim]
        """
        idx = torch.full(
            (batch_size,), self.index_for(mode),
            dtype=torch.long, device=device
        )
        return self(idx)


# ─────────────────────────────────────────────────────────────────────────────
# Frequency embedding
# ─────────────────────────────────────────────────────────────────────────────

class FrequencyEmbedding(nn.Module):
    """
    Log-sinusoidal frequency embedding.

    Encodes the centre frequency using sinusoidal features on the log scale,
    then projects to a dense vector.  This gives the model frequency priors
    without memorizing specific frequencies.

    The log scale is important: 100 Hz → 100 kHz spans the same "perceptual"
    range as 100 kHz → 100 MHz on a logarithmic frequency axis.

    Parameters
    ----------
    embed_dim    : int   — Output embedding dimension.
    num_sin_feats: int   — Number of sinusoidal frequency features.
    min_freq_hz  : float — Minimum expected centre frequency.
    max_freq_hz  : float — Maximum expected centre frequency.
    """

    def __init__(
        self,
        embed_dim:     int   = 64,
        num_sin_feats: int   = 32,
        min_freq_hz:   float = 100e3,
        max_freq_hz:   float = 6.0e9,
    ) -> None:
        super().__init__()
        self.embed_dim     = embed_dim
        self.num_sin_feats = num_sin_feats
        self.log_min       = math.log(min_freq_hz)
        self.log_max       = math.log(max_freq_hz)

        # Learnable projection from sinusoidal features → embed_dim
        self.proj = nn.Sequential(
            nn.Linear(num_sin_feats * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def _log_normalize(self, freq_hz: torch.Tensor) -> torch.Tensor:
        """Normalize frequency to [0, 1] on log scale."""
        log_f = torch.log(freq_hz.float().clamp(min=1.0))
        return (log_f - self.log_min) / (self.log_max - self.log_min)

    def forward(self, center_freq_hz: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        center_freq_hz : Tensor  [B]  — Centre frequency in Hz

        Returns
        -------
        Tensor  [B, embed_dim]
        """
        f_norm = self._log_normalize(center_freq_hz)   # [B], in [0, 1]

        # Sinusoidal encoding at multiple "octave" scales
        scales = torch.linspace(
            0.0, math.pi * self.num_sin_feats,
            self.num_sin_feats,
            device=center_freq_hz.device,
        )
        # [B, num_sin_feats]
        f_scaled = f_norm.unsqueeze(1) * scales.unsqueeze(0)
        sin_feats = torch.cat([torch.sin(f_scaled), torch.cos(f_scaled)], dim=1)

        return self.proj(sin_feats)   # [B, embed_dim]


# ─────────────────────────────────────────────────────────────────────────────
# Combined conditioning embedding
# ─────────────────────────────────────────────────────────────────────────────

class ConditioningEmbedding(nn.Module):
    """
    Combined modulation + frequency conditioning embedding.

    Fuses the modulation embedding and frequency embedding into a single
    conditioning vector used by all FiLM layers in the backbone.

    Parameters
    ----------
    mod_embed_dim  : int — Modulation embedding dimension.
    freq_embed_dim : int — Frequency embedding dimension.
    out_dim        : int — Output conditioning vector dimension.
    """

    def __init__(
        self,
        mod_embed_dim:  int = 128,
        freq_embed_dim: int = 64,
        out_dim:        int = 192,
    ) -> None:
        super().__init__()
        self.mod_embed  = ModulationEmbedding(embed_dim=mod_embed_dim)
        self.freq_embed = FrequencyEmbedding(embed_dim=freq_embed_dim)

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(mod_embed_dim + freq_embed_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    @property
    def out_dim(self) -> int:
        return self.fusion[-2].out_features

    def forward(
        self,
        mod_indices:    torch.Tensor,
        center_freq_hz: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        mod_indices    : Tensor  [B]  — modulation indices
        center_freq_hz : Tensor  [B]  — centre frequencies in Hz

        Returns
        -------
        Tensor  [B, out_dim]  — fused conditioning vector
        """
        mod_emb  = self.mod_embed(mod_indices)     # [B, mod_embed_dim]
        freq_emb = self.freq_embed(center_freq_hz) # [B, freq_embed_dim]
        fused    = torch.cat([mod_emb, freq_emb], dim=1)
        return self.fusion(fused)                  # [B, out_dim]


# ─────────────────────────────────────────────────────────────────────────────
# FiLM generator (used inside ResNet blocks and Transformer layers)
# ─────────────────────────────────────────────────────────────────────────────

class FiLMGenerator(nn.Module):
    """
    Generates FiLM scale (γ) and shift (β) parameters from a conditioning vector.

    Each residual block or Transformer layer gets its own FiLMGenerator,
    projecting from the shared conditioning embedding to that layer's channel
    dimension.  All generators share the same input but have independent
    projection weights.

    Parameters
    ----------
    cond_dim    : int — Dimension of the conditioning vector (from ConditioningEmbedding).
    feature_dim : int — Number of channels/features to modulate.
    """

    def __init__(self, cond_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.proj = nn.Linear(cond_dim, feature_dim * 2, bias=True)

        # Initialize to identity: γ=1, β=0
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias[:feature_dim], 1.0)
        nn.init.zeros_(self.proj.bias[feature_dim:])

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply FiLM modulation.

        Parameters
        ----------
        x    : Tensor  [B, C, T] or [B, T, C]  — feature map to modulate
        cond : Tensor  [B, cond_dim]            — conditioning vector

        Returns
        -------
        Tensor  same shape as x
        """
        params = self.proj(cond)                          # [B, 2*C]
        gamma  = params[:, :self.feature_dim]             # [B, C]
        beta   = params[:, self.feature_dim:]             # [B, C]

        if x.dim() == 3 and x.shape[1] == self.feature_dim:
            # Conv format: [B, C, T]
            gamma = gamma.unsqueeze(-1)                   # [B, C, 1]
            beta  = beta.unsqueeze(-1)                    # [B, C, 1]
        elif x.dim() == 3 and x.shape[2] == self.feature_dim:
            # Transformer format: [B, T, C]
            gamma = gamma.unsqueeze(1)                    # [B, 1, C]
            beta  = beta.unsqueeze(1)                     # [B, 1, C]
        else:
            # 2D: [B, C]
            pass   # gamma/beta are already [B, C]

        return gamma * x + beta
