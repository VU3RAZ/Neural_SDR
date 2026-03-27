"""
dsp/iq_utils.py
===============
IQ file I/O, format detection, and numpy ↔ torch conversion helpers.

These utilities support the ingestion and recording layers.
They are NOT part of the neural signal processing path.
"""

from __future__ import annotations

import os
import struct
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Format detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_iq_format(path: str) -> str:
    """
    Detect IQ file format from extension and magic bytes.

    Returns
    -------
    str  One of: 'sigmf' | 'wav' | 'cfile' | 'npy' | 'unknown'
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".sigmf-meta" or p.with_suffix(".sigmf-meta").exists():
        return "sigmf"
    if ext in (".wav", ".wave"):
        return "wav"
    if ext == ".npy":
        return "npy"
    if ext in (".cfile", ".iq", ".bin", ".dat", ".raw", ".cf32"):
        return "cfile"

    # Magic byte detection for extension-less files
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic[:2] == b"\x93N":   # NumPy magic
            return "npy"
        if magic == b"RIFF":         # WAV
            return "wav"
    except OSError:
        pass

    return "cfile"   # Default: treat as raw float32


# ─────────────────────────────────────────────────────────────────────────────
# IQ record writer
# ─────────────────────────────────────────────────────────────────────────────

class IQRecorder:
    """
    Streams IQ samples to a raw .cfile (interleaved float32 I/Q).

    Usage
    -----
    with IQRecorder("capture.cfile") as rec:
        rec.write(iq_chunk)
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._fp = None
        self._samples_written = 0
        self._start_time: Optional[float] = None

    def open(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._fp = open(self.path, "wb")
        self._start_time = time.monotonic()
        logger.info(f"[IQRecorder] Recording to {self.path!r}")

    def close(self) -> None:
        if self._fp:
            self._fp.flush()
            self._fp.close()
            self._fp = None
            elapsed = time.monotonic() - (self._start_time or 0)
            logger.info(
                f"[IQRecorder] Closed {self.path!r} — "
                f"{self._samples_written:,} samples in {elapsed:.1f}s"
            )

    def write(self, iq: np.ndarray) -> None:
        if self._fp is None:
            return
        iq = np.asarray(iq, dtype=np.complex64)
        # Interleave I and Q as float32
        interleaved = np.empty(len(iq) * 2, dtype=np.float32)
        interleaved[0::2] = iq.real
        interleaved[1::2] = iq.imag
        interleaved.tofile(self._fp)
        self._samples_written += len(iq)

    def __enter__(self) -> "IQRecorder":
        self.open()
        return self

    def __exit__(self, *args) -> bool:
        self.close()
        return False


# ─────────────────────────────────────────────────────────────────────────────
# SigMF metadata writer
# ─────────────────────────────────────────────────────────────────────────────

def write_sigmf_meta(
    data_path: str,
    sample_rate: float,
    center_freq: float,
    description: str = "NeuralSDR capture",
    hardware: str = "NeuralSDR",
    author: str = "NeuralSDR",
) -> str:
    """
    Write a SigMF .sigmf-meta companion file for a .cfile recording.

    Parameters
    ----------
    data_path   : str — Path to the .sigmf-data or .cfile (without .meta extension).
    sample_rate : float
    center_freq : float
    description : str
    hardware    : str
    author      : str

    Returns
    -------
    str — Path to the written .sigmf-meta file.
    """
    import json

    meta_path = str(data_path).replace(".cfile", "").replace(".sigmf-data", "")
    meta_path = meta_path + ".sigmf-meta"

    meta = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": sample_rate,
            "core:description": description,
            "core:author": author,
            "core:hw": hardware,
            "core:version": "1.0.0",
        },
        "captures": [
            {
                "core:sample_start": 0,
                "core:frequency": center_freq,
                "core:datetime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        ],
        "annotations": [],
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"[iq_utils] Wrote SigMF metadata to {meta_path!r}")
    return meta_path


# ─────────────────────────────────────────────────────────────────────────────
# NumPy ↔ PyTorch helpers
# ─────────────────────────────────────────────────────────────────────────────

def iq_to_tensor(iq: np.ndarray, device=None) -> "torch.Tensor":
    """
    Convert complex64 numpy IQ to a float32 tensor [2, N] (I and Q stacked).

    The neural models expect shape [batch, 2, chunk_size].

    Parameters
    ----------
    iq     : np.ndarray  complex64  shape (N,)
    device : torch.device or None

    Returns
    -------
    torch.Tensor  float32  shape (2, N)
    """
    import torch
    iq = np.asarray(iq, dtype=np.complex64)
    stacked = np.stack([iq.real, iq.imag], axis=0)   # (2, N)
    t = torch.from_numpy(stacked)
    if device is not None:
        t = t.to(device)
    return t


def tensor_to_iq(t: "torch.Tensor") -> np.ndarray:
    """
    Convert a float32 tensor [2, N] or [batch, 2, N] back to complex64 numpy.

    Parameters
    ----------
    t : torch.Tensor  float32  shape (2, N) or (B, 2, N)

    Returns
    -------
    np.ndarray  complex64  shape (N,) or (B, N)
    """
    arr = t.detach().cpu().numpy()
    if arr.ndim == 2:
        return (arr[0] + 1j * arr[1]).astype(np.complex64)
    elif arr.ndim == 3:
        return (arr[:, 0, :] + 1j * arr[:, 1, :]).astype(np.complex64)
    raise ValueError(f"Expected 2-D or 3-D tensor, got shape {arr.shape}")


def chunk_iq(
    iq: np.ndarray,
    chunk_size: int,
    hop_size: Optional[int] = None,
    pad: bool = True,
) -> np.ndarray:
    """
    Slice IQ into overlapping chunks.

    Parameters
    ----------
    iq         : np.ndarray  complex64  shape (N,)
    chunk_size : int
    hop_size   : int  (default = chunk_size, no overlap)
    pad        : bool  pad last chunk with zeros if needed

    Returns
    -------
    np.ndarray  complex64  shape (num_chunks, chunk_size)
    """
    if hop_size is None:
        hop_size = chunk_size

    iq = np.asarray(iq, dtype=np.complex64)
    n = len(iq)

    if pad:
        n_chunks = max(1, int(np.ceil((n - chunk_size) / hop_size)) + 1)
        padded_len = (n_chunks - 1) * hop_size + chunk_size
        iq = np.pad(iq, (0, max(0, padded_len - n)))
    else:
        n_chunks = max(0, (n - chunk_size) // hop_size + 1)

    chunks = np.lib.stride_tricks.as_strided(
        iq,
        shape=(n_chunks, chunk_size),
        strides=(iq.strides[0] * hop_size, iq.strides[0]),
    ).copy()

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalise_iq(iq: np.ndarray, target_rms: float = 1.0) -> np.ndarray:
    """
    Normalise IQ to a target RMS level.

    Parameters
    ----------
    iq         : np.ndarray  complex64
    target_rms : float

    Returns
    -------
    np.ndarray  complex64
    """
    iq = np.asarray(iq, dtype=np.complex64)
    rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
    if rms < 1e-9:
        return iq
    return (iq * (target_rms / rms)).astype(np.complex64)


def estimate_snr_db(iq: np.ndarray, noise_floor_db: float = -40.0) -> float:
    """
    Crude SNR estimate: signal power vs. assumed noise floor.

    Returns
    -------
    float  dB
    """
    from .fft_utils import signal_strength_dbfs
    signal_db = signal_strength_dbfs(iq)
    return signal_db - noise_floor_db
