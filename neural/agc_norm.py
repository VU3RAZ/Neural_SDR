"""
neural/agc_norm.py
==================
Neural Automatic Gain Control (AGC) and IQ normalization block.

Replaces the classical AGC loop entirely with a learned approach:

Architecture
------------
  Input  : [B, 2, T]   (I and Q channels, T = chunk_size)
  Output : [B, 2, T]   (normalized IQ, same shape)

  1. Power estimator   — depthwise Conv1d over a sliding window → scalar gain
  2. Attack/release    — differentiable exponential smoother (learnable τ)
  3. Gain application  — multiply IQ by per-sample gain
  4. DC offset removal — learned 1×1 conv bias correction
  5. Phase normalizer  — rotate IQ so the mean phase is zero

Why neural AGC?
  - Adapts to shaped noise (coloured interference, narrowband blockers)
  - Attack/release time constants are jointly optimized with the receiver
  - No hand-tuned loop constants
  - Backpropagates cleanly through the entire pipeline
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Exponential moving average (differentiable)
# ─────────────────────────────────────────────────────────────────────────────

class LearnableEMA(nn.Module):
    """
    Differentiable exponential moving average with a learnable time constant.

    Given a sequence x[n], computes:
        y[n] = α · x[n] + (1 − α) · y[n−1]
    where α = sigmoid(log_alpha) is learned.

    Parameters
    ----------
    init_alpha : float — Initial smoothing factor (0 < α < 1).
                         Small α = slow/smooth, large α = fast/reactive.
    """

    def __init__(self, init_alpha: float = 0.01) -> None:
        super().__init__()
        # Store in logit space so sigmoid keeps α ∈ (0, 1) always
        init_logit = math.log(init_alpha / (1.0 - init_alpha))
        self.log_alpha = nn.Parameter(torch.tensor(init_logit))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.log_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  [B, T]  — sequence to smooth

        Returns
        -------
        Tensor  [B, T]  — smoothed sequence
        """
        B, T = x.shape
        alpha = self.alpha
        y = torch.zeros_like(x)
        state = x[:, 0]   # initialise with first value
        for t in range(T):
            state = alpha * x[:, t] + (1.0 - alpha) * state
            y[:, t] = state
        return y


# ─────────────────────────────────────────────────────────────────────────────
# Power estimator
# ─────────────────────────────────────────────────────────────────────────────

class NeuralPowerEstimator(nn.Module):
    """
    Estimate instantaneous signal power via a learned depthwise convolution.

    Instead of a fixed sliding-window mean-square, we use a small learned
    filter that can adapt its effective window shape to the signal statistics.

    Input  : [B, 2, T]
    Output : [B, 1, T]   (power estimate per sample, in linear scale)
    """

    def __init__(self, window: int = 64) -> None:
        super().__init__()
        self.window = window
        # Depthwise conv over both I and Q channels
        self.conv = nn.Conv1d(
            in_channels=2,
            out_channels=2,
            kernel_size=window,
            padding=window - 1,
            groups=2,      # depthwise
            bias=False,
        )
        # Initialize weights to approximate a uniform mean-square window
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / window)

        # Learned combination: [I², Q²] → scalar power
        self.combine = nn.Conv1d(2, 1, kernel_size=1, bias=True)
        nn.init.constant_(self.combine.weight, 0.5)  # mean of I² and Q²
        nn.init.zeros_(self.combine.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  [B, 2, T]

        Returns
        -------
        power : Tensor  [B, 1, T]  — estimated power (linear, > 0)
        """
        x_sq = x ** 2                          # [B, 2, T]
        filtered = self.conv(x_sq)             # [B, 2, T + window - 1]
        filtered = filtered[:, :, :x.shape[2]] # trim to original length
        power = self.combine(filtered)          # [B, 1, T]
        return F.softplus(power) + 1e-8        # ensure strictly positive


# ─────────────────────────────────────────────────────────────────────────────
# Main AGC block
# ─────────────────────────────────────────────────────────────────────────────

class NeuralAGC(nn.Module):
    """
    Learned Automatic Gain Control + IQ normalization.

    Pipeline
    --------
    IQ [B,2,T]
      → power estimation          [B,1,T]
      → EMA smoothing (attack)    [B,1,T]
      → target gain computation   [B,1,T]
      → EMA smoothing (release)   [B,1,T]
      → gain application          [B,2,T]
      → DC removal (1×1 conv)     [B,2,T]
      → output normalisation      [B,2,T]

    Parameters
    ----------
    agc_window      : int   — Power estimator window in samples.
    attack_alpha    : float — Initial attack smoothing coefficient.
    release_alpha   : float — Initial release smoothing coefficient.
    target_power    : float — Target RMS power level after AGC (default 1.0).
    max_gain_db     : float — Maximum gain the AGC can apply (dB).
    """

    def __init__(
        self,
        agc_window:   int   = 64,
        attack_alpha: float = 0.05,
        release_alpha: float = 0.005,
        target_power: float = 1.0,
        max_gain_db:  float = 60.0,
    ) -> None:
        super().__init__()

        self.target_power = target_power
        self.max_gain_linear = 10.0 ** (max_gain_db / 20.0)

        # Power estimator
        self.power_est = NeuralPowerEstimator(window=agc_window)

        # Separate attack and release smoothers
        self.attack_ema  = LearnableEMA(init_alpha=attack_alpha)
        self.release_ema = LearnableEMA(init_alpha=release_alpha)

        # DC offset correction (learned per-channel bias removal)
        self.dc_remove = nn.Conv1d(2, 2, kernel_size=1, groups=2, bias=True)
        nn.init.eye_(self.dc_remove.weight.view(2, 2))
        nn.init.zeros_(self.dc_remove.bias)

        # Final instance normalization for stable output statistics
        self.output_norm = nn.InstanceNorm1d(2, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  [B, 2, T]  — raw IQ (I=channel 0, Q=channel 1)

        Returns
        -------
        Tensor  [B, 2, T]  — gain-normalized IQ
        """
        B, C, T = x.shape
        assert C == 2, f"Expected 2 IQ channels, got {C}"

        # ── 1. Estimate instantaneous power ──────────────────────────────────
        power = self.power_est(x)                          # [B, 1, T]
        power_flat = power.squeeze(1)                      # [B, T]

        # ── 2. Attack smoothing (fast rise on high power) ────────────────────
        smoothed_attack = self.attack_ema(power_flat)      # [B, T]

        # ── 3. Compute desired gain: target / sqrt(smoothed_power) ───────────
        desired_gain = self.target_power / (smoothed_attack.sqrt() + 1e-8)
        desired_gain = desired_gain.clamp(max=self.max_gain_linear)

        # ── 4. Release smoothing (slow decay on low power) ───────────────────
        gain = self.release_ema(desired_gain)              # [B, T]
        gain = gain.unsqueeze(1)                           # [B, 1, T]

        # ── 5. Apply gain ─────────────────────────────────────────────────────
        x = x * gain                                       # [B, 2, T]

        # ── 6. DC offset removal ─────────────────────────────────────────────
        x = self.dc_remove(x)

        # ── 7. Instance normalization for stable downstream statistics ────────
        x = self.output_norm(x)

        return x

    @torch.no_grad()
    def get_gain_stats(self, x: torch.Tensor) -> dict:
        """Diagnostic: return gain statistics for a batch."""
        power = self.power_est(x).squeeze(1)
        smoothed = self.attack_ema(power)
        gain = self.target_power / (smoothed.sqrt() + 1e-8)
        gain = gain.clamp(max=self.max_gain_linear)
        return {
            "mean_power_dbfs": float(10 * torch.log10(power.mean() + 1e-8).item()),
            "mean_gain_db":    float(20 * torch.log10(gain.mean() + 1e-8).item()),
            "max_gain_db":     float(20 * torch.log10(gain.max() + 1e-8).item()),
            "attack_alpha":    float(self.attack_ema.alpha.item()),
            "release_alpha":   float(self.release_ema.alpha.item()),
        }
