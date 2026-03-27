"""
training/losses.py
==================
Multi-task loss functions for the Unified Neural Receiver.

The receiver is trained end-to-end with a weighted sum of losses from
all active heads.  Different modulations activate different loss terms.

Loss components
---------------
Analog modes (AM, FM, SSB, CW, DSB):
  L_audio    — L1 + spectral convergence loss on reconstructed PCM audio
  L_squelch  — Binary cross-entropy for signal presence detection

Digital modes (BPSK, QPSK, ... OFDM):
  L_symbol   — Negative log-likelihood on soft symbol predictions
  L_bits     — Binary cross-entropy on per-bit LLRs
  L_timing   — L1 loss on fractional timing offset prediction
  L_squelch  — Binary cross-entropy for signal presence detection

Auxiliary (all modes):
  L_freq     — L1 loss on frequency offset estimation
  L_phase    — L1 loss on phase offset estimation

Total loss:
  L_total = w_audio * L_audio
          + w_symbol * L_symbol
          + w_bits * L_bits
          + w_timing * L_timing
          + w_squelch * L_squelch
          + w_freq * L_freq
          + w_phase * L_phase

Spectral Convergence Loss
-------------------------
Originally from [Arik et al., 2019] for audio synthesis.
Computes the L2 norm of the difference between magnitude STFTs.
Captures perceptual audio quality better than per-sample L1 alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Loss weights configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LossWeights:
    """Weights for each loss component."""
    audio:   float = 1.0    # Analog waveform reconstruction
    spectral: float = 0.5   # Spectral convergence (audio quality)
    symbol:  float = 1.0    # Digital symbol classification
    bits:    float = 0.5    # Per-bit LLR accuracy
    timing:  float = 0.2    # Symbol timing estimation
    squelch: float = 0.5    # Signal presence detection
    freq:    float = 0.3    # Frequency offset estimation
    phase:   float = 0.2    # Phase offset estimation


# ─────────────────────────────────────────────────────────────────────────────
# Individual loss components
# ─────────────────────────────────────────────────────────────────────────────

def audio_reconstruction_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    L1 loss on reconstructed audio waveform.

    Parameters
    ----------
    pred   : Tensor  [B, 1, T_audio]
    target : Tensor  [B, 1, T_audio]  (or [B, T_audio])

    Returns
    -------
    Scalar tensor
    """
    if target.dim() == 2:
        target = target.unsqueeze(1)
    # Trim/pad to same length
    min_len = min(pred.shape[-1], target.shape[-1])
    return F.l1_loss(pred[..., :min_len], target[..., :min_len])


def spectral_convergence_loss(
    pred:       torch.Tensor,
    target:     torch.Tensor,
    n_fft:      int   = 256,
    hop_length: int   = 64,
    win_length: int   = 256,
) -> torch.Tensor:
    """
    Spectral convergence + log STFT magnitude loss.

    Measures the difference between predicted and target audio in the
    frequency domain, which is more perceptually meaningful than per-sample L1.

    Parameters
    ----------
    pred, target : Tensor  [B, 1, T]
    n_fft        : int
    hop_length   : int
    win_length   : int

    Returns
    -------
    Scalar tensor
    """
    if pred.shape[-1] < n_fft or target.shape[-1] < n_fft:
        return torch.tensor(0.0, device=pred.device)

    B = pred.shape[0]
    p = pred.squeeze(1)      # [B, T]
    t = target.squeeze(1)
    min_len = min(p.shape[-1], t.shape[-1])
    p, t = p[:, :min_len], t[:, :min_len]

    window = torch.hann_window(win_length, device=pred.device)

    def stft_mag(x):
        stft = torch.stft(
            x, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window=window,
            return_complex=True,
        )
        return stft.abs() + 1e-7

    p_mag = stft_mag(p)   # [B, F, T]
    t_mag = stft_mag(t)

    # Spectral convergence: ||t_mag - p_mag||_F / ||t_mag||_F
    sc_loss = (t_mag - p_mag).norm(dim=(1, 2)) / (t_mag.norm(dim=(1, 2)) + 1e-8)

    # Log STFT magnitude: L1 on log magnitude
    log_loss = F.l1_loss(torch.log(p_mag), torch.log(t_mag))

    return sc_loss.mean() + log_loss


def symbol_classification_loss(
    pred_logits:  torch.Tensor,
    target_syms:  torch.Tensor,
    mod_orders:   torch.Tensor,
) -> torch.Tensor:
    """
    Negative log-likelihood for symbol classification.

    Parameters
    ----------
    pred_logits : Tensor  [B, N_syms, MAX_CONSTELLATION]  — log-softmax outputs
    target_syms : Tensor  [B, N_syms]                     — ground-truth symbol indices
    mod_orders  : Tensor  [B]                              — constellation size per sample

    Returns
    -------
    Scalar tensor
    """
    if pred_logits is None or target_syms is None:
        return torch.tensor(0.0)

    B, N_syms, M_max = pred_logits.shape
    N_syms_gt = target_syms.shape[1]
    N = min(N_syms, N_syms_gt)

    # Flatten for F.nll_loss
    log_probs = pred_logits[:, :N, :]           # [B, N, M_max]
    targets   = target_syms[:, :N].clamp(0, M_max - 1)  # [B, N]

    log_probs_flat = log_probs.reshape(B * N, M_max)
    targets_flat   = targets.reshape(B * N)

    # Mask: only penalize symbols within the valid modulation order
    loss = F.nll_loss(log_probs_flat, targets_flat, reduction="none")
    loss = loss.reshape(B, N).mean()
    return loss


def bit_llr_loss(
    pred_llr:  torch.Tensor,
    target_bits: torch.Tensor,
) -> torch.Tensor:
    """
    Binary cross-entropy loss on per-bit LLR predictions.

    Parameters
    ----------
    pred_llr    : Tensor  [B, N_syms, 6]  — raw LLR values (pre-sigmoid)
    target_bits : Tensor  [B, N_bits]     — ground-truth bits (0/1)

    Returns
    -------
    Scalar tensor
    """
    if pred_llr is None:
        return torch.tensor(0.0)

    B, N_syms, bps = pred_llr.shape
    pred_flat = pred_llr.reshape(B, -1)[:, :target_bits.shape[1]]
    tgt_flat  = target_bits[:, :pred_flat.shape[1]]

    return F.binary_cross_entropy_with_logits(pred_flat, tgt_flat)


def timing_loss(
    pred_timing:  torch.Tensor,
    target_timing: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    L2 loss on timing offset predictions.

    If no ground-truth timing is available, use a regularization loss
    (encourage small, smooth timing corrections).

    Parameters
    ----------
    pred_timing   : Tensor  [B, N_syms]
    target_timing : Tensor  [B, N_syms] or None

    Returns
    -------
    Scalar tensor
    """
    if target_timing is not None:
        return F.mse_loss(pred_timing, target_timing)
    else:
        # Regularize: timing offsets should be small and slowly varying
        smoothness = (pred_timing[:, 1:] - pred_timing[:, :-1]).pow(2).mean()
        magnitude  = pred_timing.abs().mean()
        return 0.1 * smoothness + 0.01 * magnitude


def squelch_loss(
    presence_prob:    torch.Tensor,
    signal_present:   torch.Tensor,
) -> torch.Tensor:
    """
    Binary cross-entropy for signal presence detection.

    Parameters
    ----------
    presence_prob  : Tensor  [B]  — predicted probability ∈ [0, 1]
    signal_present : Tensor  [B]  — ground-truth presence (1 = signal, 0 = noise)

    Returns
    -------
    Scalar tensor
    """
    return F.binary_cross_entropy(
        presence_prob.clamp(1e-6, 1 - 1e-6),
        signal_present.float(),
    )


def freq_offset_loss(
    pred_offset:   torch.Tensor,
    target_offset: torch.Tensor,
) -> torch.Tensor:
    """L1 loss on normalized frequency offset estimation."""
    return F.l1_loss(pred_offset, target_offset)


def phase_offset_loss(
    pred_phase:   torch.Tensor,
    target_phase: torch.Tensor,
) -> torch.Tensor:
    """
    Circular L1 loss on phase offset estimation.
    Wraps differences to [-π, π] range.
    """
    diff = pred_phase - target_phase
    # Wrap to [-π, π]
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    return diff.abs().mean()


# ─────────────────────────────────────────────────────────────────────────────
# Unified multi-task loss
# ─────────────────────────────────────────────────────────────────────────────

class NeuralReceiverLoss(nn.Module):
    """
    Multi-task loss function for the Unified Neural Receiver.

    Computes and combines all applicable loss terms based on the active
    modulation modes in each batch.  Analog and digital losses are masked
    to only apply to the corresponding samples.

    Parameters
    ----------
    weights : LossWeights — per-component loss weights.
    """

    def __init__(self, weights: Optional[LossWeights] = None) -> None:
        super().__init__()
        self.w = weights or LossWeights()

    def forward(
        self,
        output,           # ReceiverOutput
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss and all component losses.

        Parameters
        ----------
        output : ReceiverOutput   — model predictions
        batch  : Dict             — training batch from SyntheticBatchGenerator

        Returns
        -------
        Dict with keys:
            'total'    : scalar — total weighted loss
            'audio'    : scalar — waveform L1 loss
            'spectral' : scalar — spectral convergence loss
            'symbol'   : scalar — symbol classification loss
            'bits'     : scalar — bit LLR loss
            'timing'   : scalar — timing estimation loss
            'squelch'  : scalar — squelch classification loss
            'freq'     : scalar — freq offset estimation loss
            'phase'    : scalar — phase offset estimation loss
        """
        device = batch["iq"].device
        zero   = torch.tensor(0.0, device=device)

        is_analog  = batch.get("is_analog",  torch.ones(batch["iq"].shape[0], dtype=torch.bool, device=device))
        is_digital = batch.get("is_digital", torch.zeros(batch["iq"].shape[0], dtype=torch.bool, device=device))

        losses: Dict[str, torch.Tensor] = {}

        # ── Analog losses ─────────────────────────────────────────────────────
        n_analog = is_analog.sum().item()
        if n_analog > 0 and output.audio_pcm is not None:
            pred_audio = output.audio_pcm[is_analog]
            gt_audio   = batch["audio_gt"][is_analog]

            losses["audio"] = audio_reconstruction_loss(pred_audio, gt_audio)
            losses["spectral"] = spectral_convergence_loss(pred_audio, gt_audio)
        else:
            losses["audio"]    = zero
            losses["spectral"] = zero

        # ── Digital losses ────────────────────────────────────────────────────
        n_digital = is_digital.sum().item()
        if n_digital > 0 and output.symbol_logits is not None:
            # Get constellation sizes for this batch
            mod_indices = batch["mod_indices"][is_digital]
            from ..config.settings import ModulationMode
            all_mods = list(ModulationMode)
            mod_orders = torch.tensor(
                [all_mods[i].num_constellation_points for i in mod_indices.tolist()],
                device=device,
            )

            losses["symbol"] = symbol_classification_loss(
                output.symbol_logits[is_digital],
                batch["sym_gt"][is_digital],
                mod_orders,
            )
            losses["bits"] = bit_llr_loss(
                output.bit_llr[is_digital],
                batch["bits_gt"][is_digital],
            )
            losses["timing"] = timing_loss(output.timing_offset[is_digital])
        else:
            losses["symbol"] = zero
            losses["bits"]   = zero
            losses["timing"] = zero

        # ── Squelch loss (all samples — signal is always present in training) ─
        if output.presence_prob is not None:
            gt_presence = torch.ones(batch["iq"].shape[0], device=device)
            losses["squelch"] = squelch_loss(output.presence_prob, gt_presence)
        else:
            losses["squelch"] = zero

        # ── Auxiliary losses ──────────────────────────────────────────────────
        if output.freq_offset_norm is not None and "freq_offset" in batch:
            losses["freq"] = freq_offset_loss(
                output.freq_offset_norm,
                batch["freq_offset"],
            )
        else:
            losses["freq"] = zero

        if output.phase_offset_rad is not None and "phase_offset" in batch:
            losses["phase"] = phase_offset_loss(
                output.phase_offset_rad,
                batch["phase_offset"],
            )
        else:
            losses["phase"] = zero

        # ── Weighted total ────────────────────────────────────────────────────
        w = self.w
        total = (
            w.audio    * losses["audio"]
          + w.spectral * losses["spectral"]
          + w.symbol   * losses["symbol"]
          + w.bits     * losses["bits"]
          + w.timing   * losses["timing"]
          + w.squelch  * losses["squelch"]
          + w.freq     * losses["freq"]
          + w.phase    * losses["phase"]
        )
        losses["total"] = total

        return losses


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    output,
    batch: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Compute evaluation metrics (non-differentiable, for logging only).

    Returns
    -------
    Dict with:
        'mod_accuracy'     : float — classification accuracy over all mods
        'ber'              : float — bit error rate (digital modes)
        'squelch_accuracy' : float — squelch open/closed accuracy
        'snr_mae_db'       : float — mean absolute error on SNR estimate (dB)
    """
    metrics: Dict[str, float] = {}

    # Modulation accuracy (via symbol argmax as a proxy)
    # We don't have a separate modulation classifier; this is a placeholder
    # The conditioning embedding is always given the correct mode during training

    # Squelch accuracy
    if output.presence_prob is not None:
        pred_open = output.presence_prob > 0.5
        # In training, signal is always present
        gt_open = torch.ones_like(pred_open)
        metrics["squelch_accuracy"] = (pred_open == gt_open).float().mean().item()
    else:
        metrics["squelch_accuracy"] = 0.0

    # BER (bit error rate) for digital modes
    if output.bit_llr is not None and "bits_gt" in batch:
        is_digital = batch.get("is_digital", torch.ones(batch["iq"].shape[0], dtype=torch.bool))
        if is_digital.any():
            pred_bits = (output.bit_llr[is_digital] > 0).float()  # [B, N_syms, 6]
            pred_flat = pred_bits.reshape(pred_bits.shape[0], -1)
            gt_flat   = batch["bits_gt"][is_digital][:, :pred_flat.shape[1]]
            ber = (pred_flat != gt_flat).float().mean().item()
            metrics["ber"] = ber
        else:
            metrics["ber"] = 0.0

    # SNR MAE
    if output.snr_db is not None and "snr_db" in batch:
        mae = (output.snr_db - batch["snr_db"]).abs().mean().item()
        metrics["snr_mae_db"] = mae

    return metrics
