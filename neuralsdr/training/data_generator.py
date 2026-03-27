"""
training/data_generator.py
==========================
GPU-accelerated synthetic IQ data generator for training the NeuralReceiver.

Inspired by the RadioML dataset design [O'Shea et al., 2016] but implemented
entirely in PyTorch so generation runs on the GPU alongside training — no
disk I/O bottleneck.

What it generates per sample
-----------------------------
  iq          : float32  [2, chunk_size]   — I/Q channels stacked
  mod_index   : int64    scalar            — modulation class label
  snr_db      : float32  scalar            — ground-truth SNR
  freq_offset : float32  scalar            — ground-truth freq offset (norm)
  phase_offset: float32  scalar            — ground-truth phase offset (rad)
  audio_gt    : float32  [1, audio_len]    — ground-truth baseband audio (analog)
  symbols_gt  : int64    [N_syms]          — ground-truth symbol indices (digital)
  bits_gt     : float32  [N_bits]          — ground-truth bits (digital)

Augmentation pipeline (all differentiable, runs on GPU)
--------------------------------------------------------
  1. Baseband signal generation   (per modulation, see sources/synthetic_source.py)
  2. AWGN noise at random SNR
  3. Random frequency offset      (Δf/fs ~ Uniform(-max_offset, +max_offset))
  4. Random phase offset          (φ ~ Uniform(-π, π))
  5. Random phase noise           (Wiener process, σ ~ LogUniform)
  6. Random multipath fading      (2-tap Rayleigh, optional)
  7. Random IQ imbalance          (amplitude/phase mismatch, 0–5%)
  8. Random DC offset             (small bias on I and Q)
  9. Amplitude clipping           (ADC saturation simulation, optional)

The generator is used both:
  a) Online (during training): generates batches on-the-fly → zero disk I/O
  b) Offline (pre-generation): saves H5 datasets for reproducible validation sets
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.settings import ModulationMode, TrainingConfig, NUM_MODULATIONS
from ..sources.synthetic_source import (
    generate_iq,
    _awgn, _freq_offset, _phase_noise,
    _gen_am, _gen_fm, _gen_ssb, _gen_cw,
    _gen_bpsk, _gen_qpsk, _gen_psk8,
    _gen_qam, _gen_gfsk, _gen_ofdm,
)


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth signal generators (return signal + labels)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_sample_with_gt(
    mod:         ModulationMode,
    n_samples:   int,
    sample_rate: float,
    device:      torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Generate a clean baseband signal plus ground-truth labels.

    Returns
    -------
    dict with:
        'signal'   : complex64 tensor  [n_samples]  — clean baseband
        'audio_gt' : float32 tensor    [n_samples]  — GT audio (analog modes)
        'sym_gt'   : int64 tensor      [N_syms]     — GT symbol indices (digital)
        'bits_gt'  : float32 tensor    [N_bits]     — GT bits (digital)
        'sps'      : int               — samples per symbol
    """
    sps = max(4, int(sample_rate / 31_250))
    n_syms = n_samples // sps
    M = ModulationMode

    audio_gt = torch.zeros(n_samples, device=device, dtype=torch.float32)
    sym_gt   = torch.zeros(n_syms, device=device, dtype=torch.int64)
    bits_gt  = torch.zeros(n_syms * max(1, int(math.log2(max(2, mod.num_constellation_points)))),
                           device=device, dtype=torch.float32)

    if mod == M.AM:
        from ..sources.synthetic_source import _baseband_audio
        audio = _baseband_audio(n_samples, device)
        audio_gt = audio
        signal = _gen_am(n_samples, modulation_index=0.8, device=device)

    elif mod == M.FM_NB:
        from ..sources.synthetic_source import _baseband_audio
        audio = _baseband_audio(n_samples, device)
        audio_gt = audio
        signal = _gen_fm(n_samples, freq_dev_norm=0.008, device=device)

    elif mod == M.FM_WB:
        from ..sources.synthetic_source import _baseband_audio
        audio = _baseband_audio(n_samples, device)
        audio_gt = audio
        signal = _gen_fm(n_samples, freq_dev_norm=0.037, device=device)

    elif mod == M.USB:
        signal = _gen_ssb(n_samples, upper=True, device=device)
        audio_gt = signal.real

    elif mod == M.LSB:
        signal = _gen_ssb(n_samples, upper=False, device=device)
        audio_gt = signal.real

    elif mod == M.CW:
        signal = _gen_cw(n_samples, wpm=20.0, device=device)
        audio_gt = signal.real

    elif mod == M.DSB:
        from ..sources.synthetic_source import _baseband_audio
        audio = _baseband_audio(n_samples, device)
        audio_gt = audio
        signal = audio.to(torch.complex64)

    elif mod == M.BPSK:
        bits = torch.randint(0, 2, (n_syms,), device=device, dtype=torch.float32)
        bits_gt[:n_syms] = bits
        sym_gt = bits.long()
        signal = _gen_bpsk(n_samples, sps=sps, device=device)

    elif mod == M.QPSK:
        sym_indices = torch.randint(0, 4, (n_syms,), device=device)
        sym_gt = sym_indices
        bps = 2
        for b in range(bps):
            bits_gt[b::bps] = ((sym_indices >> (bps - 1 - b)) & 1).float()
        signal = _gen_qpsk(n_samples, sps=sps, device=device)

    elif mod == M.PSK8:
        sym_indices = torch.randint(0, 8, (n_syms,), device=device)
        sym_gt = sym_indices
        bps = 3
        for b in range(bps):
            bits_gt[b::bps] = ((sym_indices >> (bps - 1 - b)) & 1).float()
        signal = _gen_psk8(n_samples, sps=sps, device=device)

    elif mod == M.QAM16:
        sym_indices = torch.randint(0, 16, (n_syms,), device=device)
        sym_gt = sym_indices
        bps = 4
        for b in range(bps):
            bits_gt[b::bps] = ((sym_indices >> (bps - 1 - b)) & 1).float()
        signal = _gen_qam(n_samples, sps=sps, order=16, device=device)

    elif mod == M.QAM64:
        sym_indices = torch.randint(0, 64, (n_syms,), device=device)
        sym_gt = sym_indices
        bps = 6
        for b in range(bps):
            bits_gt[b::bps] = ((sym_indices >> (bps - 1 - b)) & 1).float()
        signal = _gen_qam(n_samples, sps=sps, order=64, device=device)

    elif mod == M.GFSK:
        bits = torch.randint(0, 2, (n_syms,), device=device, dtype=torch.float32)
        bits_gt[:n_syms] = bits
        sym_gt = bits.long()
        signal = _gen_gfsk(n_samples, sps=sps, bt=0.5, device=device)

    elif mod == M.CPFSK:
        bits = torch.randint(0, 2, (n_syms,), device=device, dtype=torch.float32)
        bits_gt[:n_syms] = bits
        sym_gt = bits.long()
        signal = _gen_gfsk(n_samples, sps=sps, bt=1.0, device=device)

    elif mod == M.OFDM:
        sym_indices = torch.randint(0, 4, (n_syms,), device=device)
        sym_gt = sym_indices
        bps = 2
        for b in range(bps):
            bits_gt[b::bps] = ((sym_indices >> (bps - 1 - b)) & 1).float()
        signal = _gen_ofdm(n_samples, n_subcarriers=64, device=device)

    else:
        signal = torch.zeros(n_samples, dtype=torch.complex64, device=device)

    return {
        "signal":   signal[:n_samples],
        "audio_gt": audio_gt[:n_samples],
        "sym_gt":   sym_gt,
        "bits_gt":  bits_gt,
        "sps":      sps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Channel augmentation
# ─────────────────────────────────────────────────────────────────────────────

def _apply_iq_imbalance(
    signal: torch.Tensor,
    amp_imbalance: float,
    phase_imbalance_rad: float,
) -> torch.Tensor:
    """
    Simulate IQ imbalance (hardware imperfection).
    I and Q channels have slightly different gains and a small phase error.
    """
    i_ch = signal.real
    q_ch = signal.imag
    # Amplitude mismatch
    i_ch = i_ch * (1.0 + amp_imbalance)
    q_ch = q_ch * (1.0 - amp_imbalance)
    # Phase mismatch: Q is rotated slightly off 90°
    q_ch = i_ch * math.sin(phase_imbalance_rad) + q_ch * math.cos(phase_imbalance_rad)
    return torch.complex(i_ch, q_ch)


def _apply_multipath(
    signal: torch.Tensor,
    n_taps: int = 2,
    max_delay_samples: int = 8,
) -> torch.Tensor:
    """
    Simple multipath fading: sum of delayed, attenuated copies.
    Rayleigh-distributed tap amplitudes.
    """
    n = len(signal)
    output = signal.clone()
    for _ in range(n_taps - 1):
        delay = torch.randint(1, max_delay_samples + 1, (1,)).item()
        amp   = torch.randn(1).abs().item() * 0.3
        phase = torch.rand(1).item() * 2 * math.pi
        tap   = amp * torch.exp(torch.tensor(1j * phase, dtype=torch.complex64, device=signal.device))
        padded = F.pad(signal, (delay, 0))[:n]
        output = output + tap * padded
    # Normalize power back to 1
    power = output.abs().pow(2).mean().clamp(min=1e-12)
    return output / power.sqrt()


def augment_signal(
    signal:         torch.Tensor,
    snr_db:         float,
    freq_offset:    float,
    phase_offset:   float,
    phase_noise_sigma: float = 0.005,
    apply_multipath:   bool  = False,
    apply_iq_imbalance: bool = True,
    apply_dc_offset:   bool  = True,
    apply_clipping:    bool  = False,
    clip_level:        float = 3.0,
) -> torch.Tensor:
    """
    Apply the full channel augmentation pipeline to a clean signal.

    Parameters
    ----------
    signal            : Tensor  complex64  [N]
    snr_db            : float
    freq_offset       : float — normalized (fraction of sample rate)
    phase_offset      : float — radians
    phase_noise_sigma : float
    apply_multipath   : bool
    apply_iq_imbalance: bool
    apply_dc_offset   : bool
    apply_clipping    : bool
    clip_level        : float — ADC clipping level (in sigma)

    Returns
    -------
    Tensor  complex64  [N]
    """
    n = len(signal)
    device = signal.device

    # 1. Multipath fading
    if apply_multipath and torch.rand(1).item() > 0.5:
        signal = _apply_multipath(signal, n_taps=2)

    # 2. IQ imbalance (hardware imperfection)
    if apply_iq_imbalance and torch.rand(1).item() > 0.3:
        amp_imb   = torch.rand(1).item() * 0.05    # 0–5%
        phase_imb = torch.rand(1).item() * 0.05    # 0–0.05 rad
        signal = _apply_iq_imbalance(signal, amp_imb, phase_imb)

    # 3. Phase noise (oscillator instability)
    if phase_noise_sigma > 0:
        signal = signal * _phase_noise(n, phase_noise_sigma, device)

    # 4. Frequency offset
    if abs(freq_offset) > 1e-9:
        signal = signal * _freq_offset(n, freq_offset, device)

    # 5. Phase offset
    phase_rot = torch.polar(
        torch.ones(1, device=device),
        torch.tensor([phase_offset], device=device)
    )
    signal = signal * phase_rot

    # 6. DC offset (receiver baseline offset)
    if apply_dc_offset and torch.rand(1).item() > 0.5:
        dc_i = (torch.rand(1, device=device) - 0.5) * 0.02
        dc_q = (torch.rand(1, device=device) - 0.5) * 0.02
        signal = signal + torch.complex(dc_i, dc_q)

    # 7. AWGN
    signal = _awgn(signal, snr_db)

    # 8. ADC clipping
    if apply_clipping and torch.rand(1).item() > 0.7:
        amp = signal.abs()
        limit = amp.std() * clip_level
        signal = signal * (amp.clamp(max=limit) / amp.clamp(min=1e-8))

    # 9. Normalize power
    power = signal.abs().pow(2).mean().clamp(min=1e-12)
    signal = signal / power.sqrt()

    return signal


# ─────────────────────────────────────────────────────────────────────────────
# Batch generator
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticBatchGenerator:
    """
    On-the-fly GPU-accelerated training batch generator.

    Generates complete batches of (IQ, labels) for all 15 modulation modes
    with random SNR, frequency offset, and channel impairments.

    Parameters
    ----------
    cfg         : TrainingConfig
    chunk_size  : int   — IQ samples per training example.
    sample_rate : float — Nominal sample rate for signal generation.
    device      : torch.device
    balanced    : bool  — If True, sample modulations uniformly (recommended).
                          If False, sample proportionally to a difficulty weight.
    """

    # Difficulty weights (higher = sample more frequently for harder mods)
    MOD_WEIGHTS = {
        ModulationMode.AM:    1.0,
        ModulationMode.FM_NB: 1.0,
        ModulationMode.FM_WB: 1.0,
        ModulationMode.USB:   1.5,
        ModulationMode.LSB:   1.5,
        ModulationMode.CW:    1.2,
        ModulationMode.DSB:   1.2,
        ModulationMode.BPSK:  1.0,
        ModulationMode.QPSK:  1.2,
        ModulationMode.PSK8:  1.5,
        ModulationMode.QAM16: 1.8,
        ModulationMode.QAM64: 2.5,
        ModulationMode.GFSK:  1.3,
        ModulationMode.CPFSK: 1.3,
        ModulationMode.OFDM:  2.0,
    }

    def __init__(
        self,
        cfg:         TrainingConfig,
        chunk_size:  int   = 1024,
        sample_rate: float = 2.048e6,
        device:      torch.device = torch.device("cpu"),
        balanced:    bool  = True,
    ) -> None:
        self.cfg         = cfg
        self.chunk_size  = chunk_size
        self.sample_rate = sample_rate
        self.device      = device
        self.balanced    = balanced
        self.all_mods    = list(ModulationMode)

        # Build sampling probability distribution
        if balanced:
            weights = torch.ones(NUM_MODULATIONS)
        else:
            weights = torch.tensor(
                [self.MOD_WEIGHTS.get(m, 1.0) for m in self.all_mods],
                dtype=torch.float32,
            )
        self.mod_probs = weights / weights.sum()

        # Audio output length (for analog GT)
        self.audio_len = chunk_size  # 1:1 mapping at this stage

    def _sample_augmentation_params(self, B: int) -> Dict[str, torch.Tensor]:
        """Sample random augmentation parameters for a batch."""
        snr_low, snr_high = self.cfg.snr_range_db
        fo_low, fo_high   = self.cfg.freq_offset_range
        return {
            "snr_db":         torch.FloatTensor(B).uniform_(snr_low, snr_high),
            "freq_offset":    torch.FloatTensor(B).uniform_(fo_low, fo_high),
            "phase_offset":   torch.FloatTensor(B).uniform_(-math.pi, math.pi),
            "phase_noise_sigma": torch.FloatTensor(B).uniform_(0.0, 0.01),
            "apply_multipath":   torch.BoolTensor(B).bernoulli_(0.3),
        }

    def generate_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Generate one training batch on the configured device.

        Returns
        -------
        Dict with keys:
            'iq'           : float32  [B, 2, chunk_size]
            'mod_indices'  : int64    [B]
            'center_freq'  : float32  [B]   — random in [100kHz, 6GHz]
            'bandwidth'    : float32  [B]   — random
            'snr_db'       : float32  [B]
            'freq_offset'  : float32  [B]
            'phase_offset' : float32  [B]
            'audio_gt'     : float32  [B, 1, audio_len]
            'sym_gt'       : int64    [B, N_syms]
            'bits_gt'      : float32  [B, N_bits]
            'is_analog'    : bool     [B]
            'is_digital'   : bool     [B]
        """
        # Sample modulation modes for this batch
        mod_idx_np = torch.multinomial(
            self.mod_probs.expand(batch_size, -1),
            num_samples=1,
            replacement=True,
        ).squeeze(1)   # [B]
        mods = [self.all_mods[i] for i in mod_idx_np.tolist()]

        aug_params = self._sample_augmentation_params(batch_size)

        # Estimate max GT sizes
        sps = max(4, int(self.sample_rate / 31_250))
        n_syms_max = self.chunk_size // sps
        bits_per_sym_max = 6   # log2(64) for 64-QAM

        # Pre-allocate output tensors
        iq_batch    = torch.zeros(batch_size, 2, self.chunk_size, dtype=torch.float32)
        audio_batch = torch.zeros(batch_size, 1, self.audio_len, dtype=torch.float32)
        sym_batch   = torch.zeros(batch_size, n_syms_max, dtype=torch.int64)
        bits_batch  = torch.zeros(batch_size, n_syms_max * bits_per_sym_max, dtype=torch.float32)
        is_analog   = torch.zeros(batch_size, dtype=torch.bool)
        is_digital  = torch.zeros(batch_size, dtype=torch.bool)

        # Generate each sample (vectorization limited by differing mod types)
        for i, mod in enumerate(mods):
            gt = _generate_sample_with_gt(mod, self.chunk_size, self.sample_rate, self.device)
            signal = gt["signal"]

            # Apply channel augmentation
            signal = augment_signal(
                signal,
                snr_db        = aug_params["snr_db"][i].item(),
                freq_offset   = aug_params["freq_offset"][i].item(),
                phase_offset  = aug_params["phase_offset"][i].item(),
                phase_noise_sigma = aug_params["phase_noise_sigma"][i].item(),
                apply_multipath = aug_params["apply_multipath"][i].item(),
            )

            # Stack I/Q
            iq_batch[i, 0] = signal.real
            iq_batch[i, 1] = signal.imag

            # Ground truth
            audio_batch[i, 0] = gt["audio_gt"][:self.audio_len]

            ns = min(len(gt["sym_gt"]), n_syms_max)
            sym_batch[i, :ns] = gt["sym_gt"][:ns]

            nb = min(len(gt["bits_gt"]), n_syms_max * bits_per_sym_max)
            bits_batch[i, :nb] = gt["bits_gt"][:nb]

            is_analog[i]  = mod.is_analog
            is_digital[i] = mod.is_digital

        # Random center frequencies (log-uniform across supported range)
        log_fc_min = math.log(100e3)
        log_fc_max = math.log(6e9)
        center_freq = torch.exp(
            torch.FloatTensor(batch_size).uniform_(log_fc_min, log_fc_max)
        )

        # Bandwidth: proportional to sample rate with some variation
        bandwidth = torch.FloatTensor(batch_size).uniform_(5e3, 500e3)

        return {
            "iq":           iq_batch.to(self.device),
            "mod_indices":  mod_idx_np.to(self.device),
            "center_freq":  center_freq.to(self.device),
            "bandwidth":    bandwidth.to(self.device),
            "snr_db":       aug_params["snr_db"].to(self.device),
            "freq_offset":  aug_params["freq_offset"].to(self.device),
            "phase_offset": aug_params["phase_offset"].to(self.device),
            "audio_gt":     audio_batch.to(self.device),
            "sym_gt":       sym_batch.to(self.device),
            "bits_gt":      bits_batch.to(self.device),
            "is_analog":    is_analog.to(self.device),
            "is_digital":   is_digital.to(self.device),
        }

    def estimate_throughput(self, n_batches: int = 5, batch_size: int = 32) -> float:
        """Benchmark generator throughput in samples/sec."""
        t0 = time.monotonic()
        for _ in range(n_batches):
            self.generate_batch(batch_size)
        elapsed = time.monotonic() - t0
        samples_per_sec = (n_batches * batch_size * self.chunk_size) / elapsed
        return samples_per_sec


# ─────────────────────────────────────────────────────────────────────────────
# Offline H5 dataset writer (for reproducible validation sets)
# ─────────────────────────────────────────────────────────────────────────────

def write_h5_dataset(
    path:         str,
    n_samples:    int,
    chunk_size:   int   = 1024,
    sample_rate:  float = 2.048e6,
    cfg:          Optional[TrainingConfig] = None,
    device:       torch.device = torch.device("cpu"),
    batch_size:   int   = 256,
) -> None:
    """
    Generate a fixed synthetic dataset and save to HDF5.

    Useful for creating reproducible validation sets that are identical
    across different training runs.

    Parameters
    ----------
    path        : str   — Output .h5 file path.
    n_samples   : int   — Total number of IQ examples.
    chunk_size  : int
    sample_rate : float
    cfg         : TrainingConfig (uses defaults if None)
    device      : torch.device
    batch_size  : int   — Generation batch size.
    """
    try:
        import h5py
    except ImportError:
        raise RuntimeError("h5py not installed.  Run: pip install h5py")

    from ..config.settings import TrainingConfig
    if cfg is None:
        cfg = TrainingConfig()

    gen = SyntheticBatchGenerator(
        cfg=cfg, chunk_size=chunk_size,
        sample_rate=sample_rate, device=device,
    )

    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    sps = max(4, int(sample_rate / 31_250))
    n_syms = chunk_size // sps
    n_bits = n_syms * 6

    with h5py.File(path, "w") as f:
        # Pre-allocate datasets
        ds_iq   = f.create_dataset("iq",           (n_samples, 2, chunk_size), dtype="float32")
        ds_mod  = f.create_dataset("mod_indices",  (n_samples,),               dtype="int64")
        ds_snr  = f.create_dataset("snr_db",       (n_samples,),               dtype="float32")
        ds_fo   = f.create_dataset("freq_offset",  (n_samples,),               dtype="float32")
        ds_audio= f.create_dataset("audio_gt",     (n_samples, 1, chunk_size), dtype="float32")
        ds_sym  = f.create_dataset("sym_gt",       (n_samples, n_syms),        dtype="int64")
        ds_bits = f.create_dataset("bits_gt",      (n_samples, n_bits),        dtype="float32")

        # Store metadata
        f.attrs["chunk_size"]  = chunk_size
        f.attrs["sample_rate"] = sample_rate
        f.attrs["n_samples"]   = n_samples
        f.attrs["timestamp"]   = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        n_written = 0
        while n_written < n_samples:
            B = min(batch_size, n_samples - n_written)
            batch = gen.generate_batch(B)

            ds_iq   [n_written:n_written+B] = batch["iq"].cpu().numpy()
            ds_mod  [n_written:n_written+B] = batch["mod_indices"].cpu().numpy()
            ds_snr  [n_written:n_written+B] = batch["snr_db"].cpu().numpy()
            ds_fo   [n_written:n_written+B] = batch["freq_offset"].cpu().numpy()
            ds_audio[n_written:n_written+B] = batch["audio_gt"].cpu().numpy()
            ds_sym  [n_written:n_written+B] = batch["sym_gt"][:, :n_syms].cpu().numpy()
            ds_bits [n_written:n_written+B] = batch["bits_gt"][:, :n_bits].cpu().numpy()

            n_written += B
            if n_written % (batch_size * 10) == 0:
                pct = 100 * n_written / n_samples
                from loguru import logger
                logger.info(f"  [{pct:.0f}%] Written {n_written:,}/{n_samples:,} samples")

    from loguru import logger
    size_mb = os.path.getsize(path) / 1e6
    logger.info(f"[DataGenerator] Dataset saved → {path!r}  ({size_mb:.1f} MB)")
