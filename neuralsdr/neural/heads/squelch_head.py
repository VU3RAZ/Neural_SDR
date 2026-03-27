"""
neural/heads/squelch_head.py
============================
Neural squelch head — learned signal-presence classifier.

Replaces the classical RSSI threshold squelch with a learned binary
classifier that distinguishes "real signal" from noise floor.

Architecture
------------
  Input : [B, D, T_enc]  (encoder output)
  Output:
    presence_prob  [B]    — P(signal present), in [0, 1]
    snr_estimate   [B]    — estimated SNR in dB (auxiliary output)
    energy_profile [B, T] — per-chunk signal energy (for waterfall gating)

The squelch threshold is a user-controlled hyperparameter applied to the
presence_prob output.  Setting it to 0.0 disables squelch (always pass),
setting it to 1.0 disables output completely.

Why neural squelch?
- Learns what "a real signal" looks like for each modulation mode.
- Avoids false-open on strong interference that looks like broadband noise.
- Can distinguish co-channel interference from the desired signal.
- Threshold becomes a meaningful probability rather than an arbitrary power level.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..embeddings import FiLMGenerator


class SquelchHead(nn.Module):
    """
    Learned signal-presence classifier (squelch gate).

    Parameters
    ----------
    encoder_dim : int   — Encoder output channel dimension.
    cond_dim    : int   — Conditioning vector dimension (for FiLM).
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        cond_dim:    int = 192,
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim

        # FiLM conditioning (modulation-aware squelch)
        self.film = FiLMGenerator(cond_dim=cond_dim, feature_dim=encoder_dim)

        # Global statistics extractor
        # Learns both mean and variance of the encoder features
        self.stats_extractor = nn.Sequential(
            nn.Conv1d(encoder_dim, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(128, 64, kernel_size=1),
        )

        # After global avg + std pooling: 64 * 2 = 128
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2),   # [squelch_logit, snr_logit]
        )

        # SNR estimator branch (auxiliary, for display)
        self.snr_head = nn.Linear(64, 1)

        # Per-timestep energy profile (for waterfall gating)
        self.energy_proj = nn.Conv1d(encoder_dim, 1, kernel_size=1)

    def forward(
        self,
        x:    torch.Tensor,
        cond: torch.Tensor,
    ) -> dict:
        """
        Parameters
        ----------
        x    : Tensor  [B, D, T_enc]
        cond : Tensor  [B, cond_dim]

        Returns
        -------
        dict with keys:
            'presence_prob'  : Tensor [B]    — P(signal present)
            'snr_db'         : Tensor [B]    — estimated SNR in dB
            'energy_profile' : Tensor [B, T] — per-frame energy estimate
            'squelch_open'   : Tensor [B]    — bool, True = pass audio
        """
        # FiLM conditioning
        h = self.film(x, cond)                        # [B, D, T]

        # Per-timestep energy profile
        energy = self.energy_proj(h).squeeze(1)        # [B, T]
        energy_sigmoid = torch.sigmoid(energy)

        # Global feature statistics
        stats = self.stats_extractor(h)                # [B, 64, T]
        global_mean = stats.mean(dim=-1)               # [B, 64]
        global_std  = stats.std(dim=-1).clamp(min=1e-6) # [B, 64]
        global_feat = torch.cat([global_mean, global_std], dim=1)  # [B, 128]

        # Classification
        out = self.classifier(global_feat)             # [B, 2]
        presence_logit = out[:, 0]
        snr_raw        = out[:, 1]

        # Presence probability
        presence_prob = torch.sigmoid(presence_logit)  # [B]

        # SNR estimate: scale raw output to reasonable dB range [-20, 50]
        snr_db = snr_raw * 35.0 + 15.0               # [B], centred at 15 dB

        return {
            "presence_prob":   presence_prob,
            "snr_db":          snr_db,
            "energy_profile":  energy_sigmoid,
            "squelch_open":    presence_prob > 0.5,   # default threshold
        }

    def is_open(
        self,
        x:         torch.Tensor,
        cond:      torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Convenience method: returns a boolean mask [B] for squelch gate.

        Parameters
        ----------
        x         : Tensor  [B, D, T_enc]
        cond      : Tensor  [B, cond_dim]
        threshold : float   — Probability threshold (0=always open, 1=always closed)

        Returns
        -------
        Tensor  [B]  bool
        """
        result = self.forward(x, cond)
        return result["presence_prob"] >= threshold
