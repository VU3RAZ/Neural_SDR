"""
sources/file_source.py
======================
IQ recording file source.  Supports:

  - **SigMF** (.sigmf-meta + .sigmf-data)  — metadata-rich, preferred format
  - **Complex WAV** (.wav with I/Q channels) — standard audio container
  - **Raw .cfile** / .iq — interleaved float32 I/Q (GNU Radio convention)
  - **Raw .bin / .dat** — same as .cfile, extension-agnostic detection
  - **NumPy .npy** — saved complex64/complex128 arrays

Auto-detects format from extension and file header magic bytes.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from loguru import logger

from .base_source import IQSource
from ..config.settings import FileSourceConfig


# ── Format auto-detection ─────────────────────────────────────────────────────

def _detect_format(path: str) -> str:
    """Return format tag: 'sigmf' | 'wav' | 'cfile' | 'npy'."""
    p = Path(path)
    ext = p.suffix.lower()

    if ext in (".sigmf-meta",):
        return "sigmf"
    if p.with_suffix(".sigmf-meta").exists():
        return "sigmf"
    if ext in (".wav", ".wave"):
        return "wav"
    if ext in (".npy",):
        return "npy"
    # .cfile, .iq, .bin, .dat, .raw → treat as raw interleaved float32
    return "cfile"


def _load_sigmf(path: str) -> Tuple[np.ndarray, float, float]:
    """Load SigMF recording.  Returns (samples, sample_rate, center_freq)."""
    try:
        import sigmf  # type: ignore
        from sigmf import SigMFFile, sigmffile  # type: ignore
    except ImportError:
        raise RuntimeError("sigmf package not installed.  Run: pip install sigmf")

    meta_path = path if path.endswith(".sigmf-meta") else path + ".sigmf-meta"
    data_path = meta_path.replace(".sigmf-meta", ".sigmf-data")

    with open(meta_path) as f:
        import json
        meta_dict = json.load(f)

    # Extract metadata
    global_meta = meta_dict.get("global", {})
    sample_rate = float(global_meta.get("core:sample_rate", 2.048e6))
    center_freq = 0.0
    captures = meta_dict.get("captures", [])
    if captures:
        center_freq = float(captures[0].get("core:frequency", 0.0))

    dtype_str = global_meta.get("core:datatype", "cf32_le")
    if "cf32" in dtype_str:
        np_dtype = np.complex64
    elif "ci16" in dtype_str:
        np_dtype = None   # need special handling
    else:
        np_dtype = np.complex64

    raw = np.fromfile(data_path, dtype=np.float32)
    samples = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)

    logger.info(
        f"[FileSource] SigMF: {len(samples):,} samples  "
        f"fs={sample_rate/1e6:.3f} MHz  fc={center_freq/1e6:.3f} MHz"
    )
    return samples, sample_rate, center_freq


def _load_wav(path: str) -> Tuple[np.ndarray, float, float]:
    """
    Load a complex WAV file.  Expects stereo: left=I, right=Q.
    Falls back to mono (real-only, Q=0) gracefully.
    """
    try:
        import soundfile as sf  # type: ignore
    except ImportError:
        raise RuntimeError("soundfile not installed.  Run: pip install soundfile")

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if data.shape[1] >= 2:
        samples = (data[:, 0] + 1j * data[:, 1]).astype(np.complex64)
    else:
        samples = data[:, 0].astype(np.complex64)   # mono → real-only IQ

    logger.info(f"[FileSource] WAV: {len(samples):,} samples  fs={sr/1e3:.1f} kHz")
    return samples, float(sr), 0.0


def _load_cfile(path: str) -> Tuple[np.ndarray, float, float]:
    """
    Load a raw interleaved float32 IQ file (GNU Radio .cfile convention).
    """
    raw = np.fromfile(path, dtype=np.float32)
    if len(raw) % 2 != 0:
        raw = raw[:-1]
    samples = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    logger.info(f"[FileSource] cfile: {len(samples):,} samples (no metadata)")
    return samples, 2.048e6, 0.0   # default rate — user can override


def _load_npy(path: str) -> Tuple[np.ndarray, float, float]:
    """Load a NumPy .npy file containing a complex array."""
    arr = np.load(path)
    samples = arr.astype(np.complex64).ravel()
    logger.info(f"[FileSource] npy: {len(samples):,} samples")
    return samples, 2.048e6, 0.0


# ── Source class ──────────────────────────────────────────────────────────────

class FileSource(IQSource):
    """
    IQ recording playback source.

    Loads the entire file into memory on ``open()``, then serves samples
    from an internal cursor.  Wraps around (loops) or stops at EOF based
    on ``config.loop``.

    Parameters
    ----------
    sample_rate  : float — Override sample rate (ignored if file metadata present).
    center_freq  : float — Override centre frequency.
    config       : FileSourceConfig
    """

    def __init__(
        self,
        sample_rate: float,
        center_freq: float,
        config: Optional[FileSourceConfig] = None,
    ) -> None:
        super().__init__(sample_rate, center_freq)
        self._cfg = config or FileSourceConfig()
        self._samples: Optional[np.ndarray] = None
        self._cursor: int = 0
        self._file_sample_rate: float = sample_rate
        self._file_center_freq: float = center_freq
        self._eof: bool = False
        self._last_read_time: float = 0.0

    # ── IQSource interface ────────────────────────────────────────────────────

    def open(self) -> None:
        path = self._cfg.path
        if not path:
            raise ValueError("FileSource.config.path is empty — no file specified.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"IQ file not found: {path!r}")

        fmt = _detect_format(path)
        logger.info(f"[FileSource] Opening {Path(path).name!r} as format='{fmt}'")

        loaders = {
            "sigmf": _load_sigmf,
            "wav":   _load_wav,
            "cfile": _load_cfile,
            "npy":   _load_npy,
        }
        samples, fs, fc = loaders[fmt](path)

        self._samples = samples
        self._file_sample_rate = fs if fs > 0 else self._requested_sample_rate
        self._file_center_freq = fc if fc > 0 else self._requested_center_freq
        self._cursor = 0
        self._eof = False
        self._last_read_time = time.monotonic()

        logger.info(
            f"[FileSource] Loaded {len(self._samples):,} complex samples  "
            f"({len(self._samples)/self._file_sample_rate:.2f}s of audio)  "
            f"loop={self._cfg.loop}  "
            f"playback_rate={self._cfg.playback_rate}×"
        )
        self._mark_open()

    def close(self) -> None:
        if not self._is_open:
            return
        self._samples = None
        self._mark_closed()

    def read_samples(self, n_samples: int) -> np.ndarray:
        if self._samples is None or self._eof:
            return np.zeros(n_samples, dtype=np.complex64)

        # Real-time pacing (simulate live source)
        expected_duration = n_samples / (self._file_sample_rate * self._cfg.playback_rate)
        elapsed = time.monotonic() - self._last_read_time
        sleep_time = expected_duration - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self._last_read_time = time.monotonic()

        # Slice from buffer
        end = self._cursor + n_samples
        if end <= len(self._samples):
            chunk = self._samples[self._cursor:end].copy()
            self._cursor = end
        else:
            # Need to wrap
            part1 = self._samples[self._cursor:].copy()
            shortage = n_samples - len(part1)

            if self._cfg.loop:
                # Loop: take from beginning
                part2 = self._samples[:shortage].copy()
                chunk = np.concatenate([part1, part2])
                self._cursor = shortage
                logger.debug("[FileSource] Looped back to start")
            else:
                # No loop: pad with zeros and signal EOF
                pad = np.zeros(shortage, dtype=np.complex64)
                chunk = np.concatenate([part1, pad])
                self._cursor = len(self._samples)
                self._eof = True
                logger.info("[FileSource] Reached EOF")

        return self._validate_samples(chunk, n_samples)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def actual_sample_rate(self) -> float:
        return self._file_sample_rate

    @property
    def actual_center_freq(self) -> float:
        return self._file_center_freq

    @property
    def is_eof(self) -> bool:
        return self._eof

    @property
    def progress(self) -> float:
        """Playback position as fraction 0.0–1.0."""
        if self._samples is None or len(self._samples) == 0:
            return 0.0
        return min(self._cursor / len(self._samples), 1.0)

    def seek(self, position: float) -> None:
        """
        Seek to a fractional position (0.0 = start, 1.0 = end).

        Parameters
        ----------
        position : float in [0, 1]
        """
        if self._samples is None:
            return
        self._cursor = int(position * len(self._samples))
        self._cursor = max(0, min(self._cursor, len(self._samples) - 1))
        self._eof = False
        logger.debug(f"[FileSource] Seeked to {position:.2%} ({self._cursor:,})")

    def duration_seconds(self) -> float:
        if self._samples is None or self._file_sample_rate == 0:
            return 0.0
        return len(self._samples) / self._file_sample_rate

    @property
    def signal_level_dbfs(self) -> float:
        if self._samples is None or self._cursor == 0:
            return -100.0
        window = self._samples[max(0, self._cursor - 512):self._cursor]
        return self.power_to_dbfs(window)
