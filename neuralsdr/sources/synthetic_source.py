"""
sources/synthetic_source.py
============================
Synthetic IQ generator for testing and training.

Generates realistic, physics-grounded IQ signals for every supported
modulation mode.  Designed to be *fast* — all generation runs on PyTorch
so it can be GPU-accelerated during training data generation.

RadioML-inspired design:
  - Configurable SNR, frequency offset, phase noise, multipath
  - Returns numpy complex64 (compatible with the rest of the pipeline)
  - Can be used as a source (live stream) or as a batch generator for training

Modulations implemented
-----------------------
Analog : AM, FM_NB, FM_WB, USB, LSB, CW, DSB
Digital: BPSK, QPSK, 8PSK, 16QAM, 64QAM, GFSK, CPFSK, OFDM
"""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from .base_source import IQSource
from ..config.settings import SyntheticConfig, ModulationMode


# ─────────────────────────────────────────────────────────────────────────────
# Low-level signal generators (torch, runs on GPU if available)
# ─────────────────────────────────────────────────────────────────────────────

def _awgn(signal: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add AWGN to a complex float tensor to achieve the requested SNR."""
    power = signal.abs().pow(2).mean()
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise_power = noise_power.clamp(min=1e-12)
    noise = torch.view_as_complex(
        torch.randn(*signal.shape, 2, device=signal.device) * noise_power.sqrt().item()
    )
    return signal + noise


def _phase_noise(n: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Multiplicative phase noise (Wiener process on phase)."""
    phase_increments = torch.randn(n, device=device) * sigma
    phase = torch.cumsum(phase_increments, dim=0)
    return torch.polar(torch.ones(n, device=device), phase)


def _freq_offset(n: int, offset_norm: float, device: torch.device) -> torch.Tensor:
    """Complex frequency offset: exp(j * 2π * f_norm * n)."""
    t = torch.arange(n, dtype=torch.float32, device=device)
    phase = 2.0 * math.pi * offset_norm * t
    return torch.polar(torch.ones(n, device=device), phase)


def _baseband_audio(n: int, device: torch.device) -> torch.Tensor:
    """Synthetic speech-like baseband signal (sum of sinusoids)."""
    t = torch.arange(n, dtype=torch.float32, device=device)
    # Fundamental + harmonics at audio frequencies (normalised to sample rate 1)
    freqs = [0.01, 0.017, 0.031, 0.047, 0.062]
    weights = [1.0, 0.6, 0.4, 0.2, 0.1]
    sig = sum(w * torch.sin(2 * math.pi * f * t) for f, w in zip(freqs, weights))
    sig = sig / sig.abs().max().clamp(min=1e-6)   # normalise
    return sig.float()


def _random_bits(n_bits: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, 2, (n_bits,), device=device, dtype=torch.float32)


# ── Modulation functions ──────────────────────────────────────────────────────

def _gen_am(n: int, modulation_index: float, device: torch.device) -> torch.Tensor:
    audio = _baseband_audio(n, device)
    carrier = torch.ones(n, device=device, dtype=torch.complex64)
    am = (1.0 + modulation_index * audio).to(torch.complex64) * carrier
    return am / am.abs().max().clamp(min=1e-6)


def _gen_fm(n: int, freq_dev_norm: float, device: torch.device) -> torch.Tensor:
    """FM: integrate audio to get phase, then complex exponential."""
    audio = _baseband_audio(n, device)
    phase = 2.0 * math.pi * freq_dev_norm * torch.cumsum(audio, dim=0)
    sig = torch.polar(torch.ones(n, device=device), phase)
    return sig.to(torch.complex64)


def _gen_ssb(n: int, upper: bool, device: torch.device) -> torch.Tensor:
    """SSB via Hilbert (FFT method on the baseband signal)."""
    audio = _baseband_audio(n, device)
    audio_np = audio.cpu().numpy()
    spectrum = np.fft.rfft(audio_np, n=n)
    analytic_spec = spectrum.copy()
    if upper:
        analytic_spec[:n // 4] = 0   # zero lower sideband
    else:
        analytic_spec[n // 4:] = 0   # zero upper sideband
    analytic = np.fft.irfft(analytic_spec, n=n).astype(np.float32)
    sig = (analytic + 1j * np.imag(np.fft.ifft(
        np.concatenate([spectrum, -spectrum[::-1].conj()])
    ))).astype(np.complex64)
    return torch.from_numpy(sig[:n]).to(device)


def _gen_cw(n: int, wpm: float, device: torch.device) -> torch.Tensor:
    """CW: a single carrier gated by a random Morse-like pattern."""
    # Dot length in samples (normalised): 1200 / wpm millidots at fs=1
    dot_len = max(1, int(n / (wpm * 4)))
    carrier = torch.ones(n, dtype=torch.complex64, device=device)
    gate = torch.zeros(n, dtype=torch.float32, device=device)
    pos = 0
    while pos < n:
        element = torch.randint(0, 3, (1,)).item()
        length = dot_len * (1 if element == 0 else (3 if element == 1 else 1))
        end = min(pos + length, n)
        if element != 2:  # 0=dit, 1=dah → on; 2=space → off
            gate[pos:end] = 1.0
        pos = end
    return carrier * gate.to(torch.complex64)


def _gen_bpsk(n: int, sps: int, device: torch.device) -> torch.Tensor:
    """BPSK with pulse shaping (rectangular for simplicity)."""
    n_syms = n // sps
    bits = _random_bits(n_syms, device) * 2 - 1  # ±1
    symbols = bits.repeat_interleave(sps).to(torch.complex64)
    return symbols[:n] / math.sqrt(1.0)


def _gen_qpsk(n: int, sps: int, device: torch.device) -> torch.Tensor:
    """QPSK: 2 bits/symbol, Gray coded."""
    n_syms = n // sps
    i_bits = _random_bits(n_syms, device) * 2 - 1
    q_bits = _random_bits(n_syms, device) * 2 - 1
    symbols = (i_bits + 1j * q_bits).to(torch.complex64) / math.sqrt(2)
    return symbols.repeat_interleave(sps)[:n]


def _gen_psk8(n: int, sps: int, device: torch.device) -> torch.Tensor:
    """8-PSK: uniform phase spacing."""
    n_syms = n // sps
    idx = torch.randint(0, 8, (n_syms,), device=device)
    phases = 2 * math.pi * idx.float() / 8
    symbols = torch.polar(torch.ones(n_syms, device=device), phases).to(torch.complex64)
    return symbols.repeat_interleave(sps)[:n]


def _gen_qam(n: int, sps: int, order: int, device: torch.device) -> torch.Tensor:
    """Square QAM (16 or 64)."""
    m = int(math.sqrt(order))
    assert m * m == order
    n_syms = n // sps
    i = torch.randint(0, m, (n_syms,), device=device).float() - (m - 1) / 2
    q = torch.randint(0, m, (n_syms,), device=device).float() - (m - 1) / 2
    symbols = (i + 1j * q).to(torch.complex64)
    symbols /= symbols.abs().max().clamp(min=1e-6)
    return symbols.repeat_interleave(sps)[:n]


def _gen_gfsk(n: int, sps: int, bt: float, device: torch.device) -> torch.Tensor:
    """GFSK: Gaussian-smoothed FSK."""
    n_syms = n // sps
    bits = _random_bits(n_syms, device) * 2 - 1
    # Gaussian filter approximation (no scipy — use convolution with Gaussian kernel)
    kernel_len = sps * 4
    t = torch.linspace(-2, 2, kernel_len, device=device)
    gauss = torch.exp(-t.pow(2) / (2 * bt ** 2))
    gauss /= gauss.sum()
    # Upsample bits
    upsampled = bits.repeat_interleave(sps)[:n].float()
    # Convolve (pad for same-length output)
    pad = kernel_len // 2
    upsampled_padded = F.pad(upsampled.unsqueeze(0).unsqueeze(0), (pad, pad))
    filtered = F.conv1d(upsampled_padded, gauss.view(1, 1, -1)).squeeze()[:n]
    # FM modulate
    phase = math.pi * torch.cumsum(filtered, dim=0) / sps
    return torch.polar(torch.ones(n, device=device), phase).to(torch.complex64)


def _gen_ofdm(n: int, n_subcarriers: int, device: torch.device) -> torch.Tensor:
    """Basic OFDM with QPSK subcarriers, no guard interval (simplified)."""
    n_syms = n // n_subcarriers
    i_bits = _random_bits(n_syms * n_subcarriers, device) * 2 - 1
    q_bits = _random_bits(n_syms * n_subcarriers, device) * 2 - 1
    freq_domain = (i_bits + 1j * q_bits).view(n_syms, n_subcarriers).to(torch.complex64)
    freq_domain /= math.sqrt(2 * n_subcarriers)
    # IFFT each symbol
    time_domain = torch.fft.ifft(freq_domain, dim=1).reshape(-1)
    return time_domain[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Generator registry
# ─────────────────────────────────────────────────────────────────────────────

def generate_iq(
    modulation: ModulationMode,
    n_samples: int,
    sample_rate: float,
    snr_db: float = 15.0,
    freq_offset_norm: float = 0.0,
    phase_noise_sigma: float = 0.005,
    seed: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """
    Generate ``n_samples`` of synthetic IQ for the given modulation.

    Parameters
    ----------
    modulation        : ModulationMode
    n_samples         : int — Number of complex output samples.
    sample_rate       : float — Sample rate (Hz).  Used to scale freq deviations.
    snr_db            : float — Signal-to-noise ratio.
    freq_offset_norm  : float — Frequency offset as fraction of sample rate.
    phase_noise_sigma : float — Phase noise (rad/sample, Wiener).
    seed              : Optional[int] — RNG seed for reproducibility.
    device            : torch.device

    Returns
    -------
    np.ndarray  shape (n_samples,)  dtype complex64
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    sps = max(4, int(sample_rate / 31_250))   # ~31.25 kbaud symbol rate default

    mod = modulation
    M = ModulationMode

    if   mod == M.AM:     sig = _gen_am(n_samples, modulation_index=0.8, device=device)
    elif mod == M.FM_NB:  sig = _gen_fm(n_samples, freq_dev_norm=0.008, device=device)
    elif mod == M.FM_WB:  sig = _gen_fm(n_samples, freq_dev_norm=0.037, device=device)
    elif mod == M.USB:    sig = _gen_ssb(n_samples, upper=True, device=device)
    elif mod == M.LSB:    sig = _gen_ssb(n_samples, upper=False, device=device)
    elif mod == M.CW:     sig = _gen_cw(n_samples, wpm=20.0, device=device)
    elif mod == M.DSB:
        audio = _baseband_audio(n_samples, device)
        sig = audio.to(torch.complex64)
    elif mod == M.BPSK:   sig = _gen_bpsk(n_samples, sps=sps, device=device)
    elif mod == M.QPSK:   sig = _gen_qpsk(n_samples, sps=sps, device=device)
    elif mod == M.PSK8:   sig = _gen_psk8(n_samples, sps=sps, device=device)
    elif mod == M.QAM16:  sig = _gen_qam(n_samples, sps=sps, order=16, device=device)
    elif mod == M.QAM64:  sig = _gen_qam(n_samples, sps=sps, order=64, device=device)
    elif mod == M.GFSK:   sig = _gen_gfsk(n_samples, sps=sps, bt=0.5, device=device)
    elif mod == M.CPFSK:  sig = _gen_gfsk(n_samples, sps=sps, bt=1.0, device=device)
    elif mod == M.OFDM:   sig = _gen_ofdm(n_samples, n_subcarriers=64, device=device)
    else:
        raise ValueError(f"No generator implemented for {mod}")

    # Ensure length
    if len(sig) < n_samples:
        sig = F.pad(sig.real, (0, n_samples - len(sig))).to(torch.complex64)
    sig = sig[:n_samples]

    # Apply phase noise
    if phase_noise_sigma > 0:
        sig = sig * _phase_noise(n_samples, phase_noise_sigma, device)

    # Apply frequency offset
    if abs(freq_offset_norm) > 1e-9:
        sig = sig * _freq_offset(n_samples, freq_offset_norm, device)

    # Add AWGN
    sig = _awgn(sig, snr_db)

    # Normalise to unit power
    power = sig.abs().pow(2).mean().clamp(min=1e-12)
    sig = sig / power.sqrt()

    return sig.cpu().numpy().astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
# IQSource implementation
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticSource(IQSource):
    """
    Infinite synthetic IQ stream for testing the full pipeline.

    Generates IQ at the configured modulation / SNR and serves it
    continuously from ``read_samples()``.

    Parameters
    ----------
    sample_rate  : float
    center_freq  : float
    modulation   : ModulationMode
    config       : SyntheticConfig
    device       : torch.device — Generation device (use GPU for training).
    """

    def __init__(
        self,
        sample_rate: float,
        center_freq: float,
        modulation: ModulationMode = ModulationMode.FM_WB,
        config: Optional[SyntheticConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(sample_rate, center_freq)
        self._mod = modulation
        self._cfg = config or SyntheticConfig()
        self._device = device or torch.device("cpu")
        self._call_count: int = 0

    # ── IQSource interface ────────────────────────────────────────────────────

    def open(self) -> None:
        logger.info(
            f"[SyntheticSource] Opened — "
            f"mod={self._mod.value}  "
            f"SNR={self._cfg.snr_db:.1f} dB  "
            f"noise_floor={self._cfg.noise_floor_db:.1f} dBFS  "
            f"device={self._device}"
        )
        self._mark_open()

    def close(self) -> None:
        if not self._is_open:
            return
        self._mark_closed()

    def read_samples(self, n_samples: int) -> np.ndarray:
        seed = None
        if self._cfg.seed is not None:
            seed = self._cfg.seed + self._call_count
        self._call_count += 1

        samples = generate_iq(
            modulation=self._mod,
            n_samples=n_samples,
            sample_rate=self._requested_sample_rate,
            snr_db=self._cfg.snr_db,
            freq_offset_norm=self._cfg.freq_offset_hz / self._requested_sample_rate,
            seed=seed,
            device=self._device,
        )

        # Simulate real-time pacing (1 chunk per chunk_duration)
        time.sleep(n_samples / self._requested_sample_rate * 0.9)

        return self._validate_samples(samples, n_samples)

    # ── Configuration updates ─────────────────────────────────────────────────

    def set_modulation(self, modulation: ModulationMode) -> None:
        self._mod = modulation
        logger.debug(f"[SyntheticSource] Modulation → {modulation.value}")

    def set_snr(self, snr_db: float) -> None:
        self._cfg = SyntheticConfig(
            snr_db=snr_db,
            noise_floor_db=self._cfg.noise_floor_db,
            freq_offset_hz=self._cfg.freq_offset_hz,
            seed=self._cfg.seed,
        )

    @property
    def signal_level_dbfs(self) -> float:
        return self._cfg.noise_floor_db + self._cfg.snr_db
