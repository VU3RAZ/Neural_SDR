"""
dsp/fft_utils.py
================
FFT helpers used ONLY for spectrum and waterfall display.

These functions are intentionally kept out of the neural processing path.
They exist solely to feed the GUI with pretty visuals.

Rules:
  - Only called from gui/ modules.
  - Never imported by neural/ or sources/.
  - May use numpy/scipy for display convenience.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Window functions
# ─────────────────────────────────────────────────────────────────────────────

WINDOWS = {
    "hann":        np.hanning,
    "hamming":     np.hamming,
    "blackman":    np.blackman,
    "bartlett":    np.bartlett,
    "rectangular": np.ones,
}


def make_window(name: str, n: int) -> np.ndarray:
    factory = WINDOWS.get(name.lower(), np.hanning)
    return factory(n).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Power spectrum (for spectrum plot)
# ─────────────────────────────────────────────────────────────────────────────

def power_spectrum(
    iq: np.ndarray,
    fft_size: int = 2048,
    window: str = "hann",
    overlap: float = 0.5,
    db_ref: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a Welch-averaged power spectrum from complex IQ samples.

    Parameters
    ----------
    iq       : np.ndarray  complex64, shape (N,)
    fft_size : int         FFT length
    window   : str         Window function name
    overlap  : float       Fractional overlap (0–<1)
    db_ref   : float       Reference power for dB conversion

    Returns
    -------
    freqs_norm : np.ndarray  float32  shape (fft_size,)  normalised freq [-0.5, 0.5)
    psd_db     : np.ndarray  float32  shape (fft_size,)  power in dBFS
    """
    iq = np.asarray(iq, dtype=np.complex64)
    n = len(iq)
    hop = max(1, int(fft_size * (1.0 - overlap)))
    win = make_window(window, fft_size)
    win_power = np.sum(win ** 2)

    accum = np.zeros(fft_size, dtype=np.float64)
    count = 0

    for start in range(0, n - fft_size + 1, hop):
        frame = iq[start:start + fft_size] * win
        spectrum = np.fft.fft(frame, n=fft_size)
        accum += np.abs(spectrum) ** 2
        count += 1

    if count == 0:
        return np.linspace(-0.5, 0.5, fft_size, dtype=np.float32), \
               np.full(fft_size, -100.0, dtype=np.float32)

    psd = accum / (count * win_power)
    psd_db = 10.0 * np.log10(psd / db_ref + 1e-20)

    # FFT-shift so DC is in the centre
    psd_db = np.fft.fftshift(psd_db).astype(np.float32)
    freqs_norm = np.fft.fftshift(np.fft.fftfreq(fft_size)).astype(np.float32)

    return freqs_norm, psd_db


def power_spectrum_fast(
    iq: np.ndarray,
    fft_size: int = 2048,
    window: str = "hann",
) -> np.ndarray:
    """
    Single-shot (no averaging) power spectrum.  Faster, less smooth.

    Returns
    -------
    np.ndarray  float32  shape (fft_size,)  dBFS, DC-centred.
    """
    iq = np.asarray(iq[:fft_size], dtype=np.complex64)
    if len(iq) < fft_size:
        iq = np.pad(iq, (0, fft_size - len(iq)))
    win = make_window(window, fft_size)
    spectrum = np.fft.fftshift(np.fft.fft(iq * win))
    psd_db = 10.0 * np.log10(np.abs(spectrum) ** 2 + 1e-20).astype(np.float32)
    return psd_db


# ─────────────────────────────────────────────────────────────────────────────
# Waterfall accumulator
# ─────────────────────────────────────────────────────────────────────────────

class WaterfallAccumulator:
    """
    Ring buffer of power spectra for the waterfall display.

    Call ``push(iq_chunk)`` on each new IQ block.  Read ``image`` for the
    2-D waterfall array (newest row at top, shape = [history, fft_size]).
    """

    def __init__(
        self,
        fft_size: int = 2048,
        history: int = 200,
        window: str = "hann",
        db_min: float = -100.0,
        db_max: float = 0.0,
    ) -> None:
        self.fft_size = fft_size
        self.history = history
        self.window = window
        self.db_min = db_min
        self.db_max = db_max
        self._buf = np.full((history, fft_size), db_min, dtype=np.float32)
        self._head = 0   # next write row

    def push(self, iq: np.ndarray) -> None:
        """Add one IQ block to the waterfall buffer."""
        psd = power_spectrum_fast(iq, self.fft_size, self.window)
        self._buf[self._head] = psd
        self._head = (self._head + 1) % self.history

    @property
    def image(self) -> np.ndarray:
        """
        Return the waterfall as a 2-D array with newest row at index 0.
        Shape: (history, fft_size), dtype float32, range [db_min, db_max].
        """
        # Roll so the most recent row is at the top
        rolled = np.roll(self._buf, -self._head, axis=0)[::-1]
        # Clip to display range
        return np.clip(rolled, self.db_min, self.db_max)

    @property
    def image_normalised(self) -> np.ndarray:
        """
        Normalised waterfall in [0, 1] range for image display.
        """
        img = self.image
        span = self.db_max - self.db_min
        if span < 1e-6:
            return np.zeros_like(img)
        return (img - self.db_min) / span


# ─────────────────────────────────────────────────────────────────────────────
# Frequency axis helpers
# ─────────────────────────────────────────────────────────────────────────────

def freq_axis_mhz(
    center_freq_hz: float,
    sample_rate_hz: float,
    fft_size: int,
) -> np.ndarray:
    """
    Return frequency axis in MHz for a spectrum plot, DC-centred.

    Parameters
    ----------
    center_freq_hz : float
    sample_rate_hz : float
    fft_size       : int

    Returns
    -------
    np.ndarray  float32  shape (fft_size,)  in MHz
    """
    freqs_norm = np.fft.fftshift(np.fft.fftfreq(fft_size))
    return (center_freq_hz + freqs_norm * sample_rate_hz).astype(np.float32) / 1e6


# ─────────────────────────────────────────────────────────────────────────────
# Signal strength meter
# ─────────────────────────────────────────────────────────────────────────────

def signal_strength_dbfs(iq: np.ndarray) -> float:
    """
    Compute instantaneous signal power in dBFS.

    Parameters
    ----------
    iq : np.ndarray  complex64

    Returns
    -------
    float  dBFS  (0 dBFS = full scale, negative = below full scale)
    """
    iq = np.asarray(iq, dtype=np.complex64)
    power = float(np.mean(np.abs(iq) ** 2))
    if power <= 0:
        return -120.0
    return float(10.0 * np.log10(power))


def peak_frequency_hz(
    iq: np.ndarray,
    center_freq_hz: float,
    sample_rate_hz: float,
    fft_size: int = 2048,
) -> float:
    """
    Find the frequency of the strongest spectral peak.

    Returns
    -------
    float  Hz  (absolute frequency of the dominant component)
    """
    psd = power_spectrum_fast(iq, fft_size)
    peak_bin = int(np.argmax(psd))
    freqs = freq_axis_mhz(center_freq_hz, sample_rate_hz, fft_size) * 1e6
    return float(freqs[peak_bin])


# ─────────────────────────────────────────────────────────────────────────────
# Constellation extractor (display only)
# ─────────────────────────────────────────────────────────────────────────────

def extract_constellation_points(
    iq: np.ndarray,
    sps: int = 8,
    max_points: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample IQ to symbol rate for constellation display.

    Parameters
    ----------
    iq         : np.ndarray  complex64
    sps        : int  samples per symbol (approximate)
    max_points : int  maximum constellation points to return

    Returns
    -------
    i_points : np.ndarray  float32
    q_points : np.ndarray  float32
    """
    iq = np.asarray(iq, dtype=np.complex64)
    # Simple symbol-rate downsampling
    decimated = iq[::sps]
    decimated = decimated[:max_points]
    # Normalise to unit circle for display
    scale = float(np.sqrt(np.mean(np.abs(decimated) ** 2))) or 1.0
    decimated /= scale
    return decimated.real.astype(np.float32), decimated.imag.astype(np.float32)
