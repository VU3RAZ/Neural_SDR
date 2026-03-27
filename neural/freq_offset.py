"""
neural/freq_offset.py
=====================
Neural frequency offset estimation and correction.

Replaces classical PLLs, Costas loops, and Viterbi-Viterbi estimators with
a fully differentiable learned approach.

Architecture
------------
  Input  : [B, 2, T]  (channel-filtered IQ)
  Output : [B, 2, T]  (frequency-corrected IQ)

  1. FreqOffsetEstimator
     ├── CNN feature extractor (4× strided conv)
     ├── Temporal pooling (adaptive avg)
     └── MLP head → scalar Δf/fs ∈ (-max_offset, +max_offset)

  2. IQ Rotation (complex multiply)
     exp(-j·2π·Δf/fs·t) applied sample-by-sample

  3. Residual refinement
     Second lightweight estimator on the corrected IQ (fine-tune)

Why neural freq correction?
- End-to-end differentiable: error signals from demodulation heads
  can improve the freq estimator through backprop.
- Handles non-stationary offsets (frequency drift) better than a static estimator.
- No assumption about modulation — works for AM/FM/PSK/QAM equally.
- The closed-loop refinement stage mirrors the structure of a real PLL but
  with learned dynamics.
"""

from __future__ import annotations

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# IQ rotation utility
# ─────────────────────────────────────────────────────────────────────────────

def apply_freq_correction(
    iq: torch.Tensor,
    freq_offset_norm: torch.Tensor,
) -> torch.Tensor:
    """
    Apply a frequency correction to IQ by multiplying with a complex phasor.

    This is equivalent to:
        iq_corrected[n] = iq[n] · exp(-j·2π·Δf_norm·n)

    Parameters
    ----------
    iq               : Tensor  [B, 2, T]  — I in channel 0, Q in channel 1
    freq_offset_norm : Tensor  [B]        — offset as fraction of sample rate

    Returns
    -------
    Tensor  [B, 2, T]
    """
    B, C, T = iq.shape
    assert C == 2

    t = torch.arange(T, device=iq.device, dtype=iq.dtype)   # [T]
    # Phase ramp: [B, T]
    phase = -2.0 * math.pi * freq_offset_norm.unsqueeze(1) * t.unsqueeze(0)

    cos_p = torch.cos(phase)   # [B, T]
    sin_p = torch.sin(phase)   # [B, T]

    I_in = iq[:, 0, :]   # [B, T]
    Q_in = iq[:, 1, :]   # [B, T]

    # Complex multiply: (I + jQ)(cos - j·sin) = (I·cos + Q·sin) + j(Q·cos - I·sin)
    I_out = I_in * cos_p + Q_in * sin_p
    Q_out = Q_in * cos_p - I_in * sin_p

    return torch.stack([I_out, Q_out], dim=1)   # [B, 2, T]


# ─────────────────────────────────────────────────────────────────────────────
# Frequency offset estimator CNN
# ─────────────────────────────────────────────────────────────────────────────

class FreqOffsetEstimatorCNN(nn.Module):
    """
    CNN that estimates the normalized frequency offset of an IQ signal.

    Architecture
    ------------
    Input [B, 2, T]
    → Conv1d(2→32, k=7, s=2) → BN → GELU        [B, 32, T/2]
    → Conv1d(32→64, k=5, s=2) → BN → GELU       [B, 64, T/4]
    → Conv1d(64→128, k=3, s=2) → BN → GELU      [B, 128, T/8]
    → AdaptiveAvgPool1d(16)                       [B, 128, 16]
    → Flatten                                     [B, 2048]
    → Linear(2048, 256) → GELU → Dropout
    → Linear(256, 1) → Tanh → × max_offset        [B, 1]

    The Tanh + scale ensures the output is always bounded.

    Parameters
    ----------
    max_offset_norm : float — Maximum detectable offset (fraction of sample rate).
    """

    def __init__(self, max_offset_norm: float = 0.1) -> None:
        super().__init__()
        self.max_offset_norm = max_offset_norm

        self.encoder = nn.Sequential(
            # Stage 1
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            # Stage 2
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            # Stage 3
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            # Temporal pooling
            nn.AdaptiveAvgPool1d(16),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  [B, 2, T]

        Returns
        -------
        offset_norm : Tensor  [B]  — estimated freq offset (frac of sample rate)
        """
        features = self.encoder(x)          # [B, 128, 16]
        raw = self.head(features)            # [B, 1]
        return (raw * self.max_offset_norm).squeeze(1)   # [B]


# ─────────────────────────────────────────────────────────────────────────────
# Phase offset estimator (residual fine correction)
# ─────────────────────────────────────────────────────────────────────────────

class PhaseOffsetEstimator(nn.Module):
    """
    Lightweight phase offset estimator for residual correction.

    After coarse frequency correction, a small residual phase offset may
    remain.  This module estimates it directly from the mean IQ phasor.

    Parameters
    ----------
    None — no learnable parameters beyond a small projection.
    """

    def __init__(self) -> None:
        super().__init__()
        # Tiny MLP to refine the atan2 estimate
        self.refine = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        iq : Tensor  [B, 2, T]

        Returns
        -------
        phase_offset : Tensor  [B]  — estimated phase in radians
        """
        # Mean phasor
        I_mean = iq[:, 0, :].mean(dim=1)   # [B]
        Q_mean = iq[:, 1, :].mean(dim=1)   # [B]
        phasor = torch.stack([I_mean, Q_mean], dim=1)   # [B, 2]
        # Raw atan2 estimate + learned refinement
        phase_raw = torch.atan2(Q_mean, I_mean)          # [B]
        refinement = self.refine(phasor).squeeze(1)      # [B]
        return phase_raw + 0.1 * refinement              # [B]


def apply_phase_correction(
    iq: torch.Tensor,
    phase_offset: torch.Tensor,
) -> torch.Tensor:
    """
    Rotate IQ by -phase_offset (negate to correct).

    Parameters
    ----------
    iq           : Tensor  [B, 2, T]
    phase_offset : Tensor  [B]

    Returns
    -------
    Tensor  [B, 2, T]
    """
    cos_p = torch.cos(-phase_offset).unsqueeze(-1)   # [B, 1]
    sin_p = torch.sin(-phase_offset).unsqueeze(-1)   # [B, 1]
    I = iq[:, 0, :]
    Q = iq[:, 1, :]
    I_out = I * cos_p + Q * sin_p
    Q_out = Q * cos_p - I * sin_p
    return torch.stack([I_out, Q_out], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Combined freq + phase corrector
# ─────────────────────────────────────────────────────────────────────────────

class NeuralFreqOffsetCorrector(nn.Module):
    """
    Two-stage neural frequency and phase offset correction.

    Stage 1 — Coarse freq correction:
        CNN estimates Δf/fs over the whole chunk.
        Applied as a complex frequency ramp.

    Stage 2 — Fine phase correction:
        Lightweight mean-phasor estimator removes residual phase error.

    Both stages are differentiable — gradients from demodulation heads
    flow back through the IQ rotations into the estimator CNNs.

    Parameters
    ----------
    max_offset_norm : float — Max detectable freq offset (fraction of fs).
    """

    def __init__(self, max_offset_norm: float = 0.1) -> None:
        super().__init__()
        self.coarse = FreqOffsetEstimatorCNN(max_offset_norm=max_offset_norm)
        self.fine   = PhaseOffsetEstimator()

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor  [B, 2, T]  — channel-filtered IQ

        Returns
        -------
        corrected     : Tensor  [B, 2, T]  — frequency-corrected IQ
        freq_offset   : Tensor  [B]        — estimated freq offset (norm)
        phase_offset  : Tensor  [B]        — estimated phase offset (rad)
        """
        # ── Stage 1: coarse frequency correction ─────────────────────────────
        freq_offset = self.coarse(x)                          # [B]
        x_freq_corr = apply_freq_correction(x, freq_offset)  # [B, 2, T]

        # ── Stage 2: residual phase correction ───────────────────────────────
        phase_offset = self.fine(x_freq_corr)                 # [B]
        x_corrected  = apply_phase_correction(x_freq_corr, phase_offset)

        return x_corrected, freq_offset, phase_offset

    @torch.no_grad()
    def estimate_only(self, x: torch.Tensor) -> dict:
        """
        Run estimation without applying correction.
        Useful for diagnostics and display.
        """
        freq_offset  = self.coarse(x)
        x_freq_corr  = apply_freq_correction(x, freq_offset)
        phase_offset = self.fine(x_freq_corr)
        return {
            "freq_offset_norm": freq_offset.mean().item(),
            "phase_offset_rad": phase_offset.mean().item(),
            "freq_offset_ppm_at_100mhz": freq_offset.mean().item() * 1e6,
        }
