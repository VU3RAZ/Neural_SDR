"""
sources/base_source.py
======================
Abstract base class that every IQ source must implement.

Design principles
-----------------
- ``read_samples()``  is the single required method — returns a numpy complex64 array.
- Sources are context managers (``__enter__`` / ``__exit__``) so resources are
  always released, even on exceptions.
- Every source exposes ``is_open`` and ``actual_sample_rate`` properties so the
  rest of the pipeline can query real vs. requested rates.
- Async variants (``aread_samples``) are provided for sources that benefit from
  non-blocking I/O (WebSDR).  Synchronous sources simply wrap the sync version.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from loguru import logger


class IQSource(ABC):
    """
    Abstract base class for all NeuralSDR IQ acquisition sources.

    Subclasses MUST implement:
      - ``open()``
      - ``close()``
      - ``read_samples(n_samples)`` → np.ndarray[complex64]

    Subclasses SHOULD implement:
      - ``actual_sample_rate`` property
      - ``signal_level_dbfs`` property (for the strength meter)
    """

    def __init__(self, sample_rate: float, center_freq: float) -> None:
        self._requested_sample_rate: float = sample_rate
        self._requested_center_freq: float = center_freq
        self._is_open: bool = False
        self._total_samples_read: int = 0
        self._open_time: Optional[float] = None

    # ── Required interface ────────────────────────────────────────────────────

    @abstractmethod
    def open(self) -> None:
        """Open the hardware / connection / file.  Must set ``self._is_open``."""

    @abstractmethod
    def close(self) -> None:
        """Release all resources.  Must clear ``self._is_open``."""

    @abstractmethod
    def read_samples(self, n_samples: int) -> np.ndarray:
        """
        Return exactly ``n_samples`` complex64 IQ samples.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples,)``, dtype ``complex64``.
            I = real part, Q = imaginary part.
        """

    # ── Optional overrides ────────────────────────────────────────────────────

    @property
    def actual_sample_rate(self) -> float:
        """Return the hardware/stream sample rate (may differ from requested)."""
        return self._requested_sample_rate

    @property
    def actual_center_freq(self) -> float:
        """Return the hardware centre frequency."""
        return self._requested_center_freq

    @property
    def signal_level_dbfs(self) -> float:
        """
        Instantaneous signal power in dBFS.
        Override in subclasses that have hardware RSSI.
        Default: -inf (unknown).
        """
        return float("-inf")

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def source_name(self) -> str:
        return self.__class__.__name__

    # ── Async wrapper ─────────────────────────────────────────────────────────

    async def aread_samples(self, n_samples: int) -> np.ndarray:
        """
        Async wrapper around ``read_samples``.
        Sources with native async support (e.g. WebSDR) should override this.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.read_samples, n_samples)

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "IQSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False  # do not suppress exceptions

    # ── Utility helpers ───────────────────────────────────────────────────────

    def _mark_open(self) -> None:
        self._is_open = True
        self._open_time = time.monotonic()
        logger.info(f"[{self.source_name}] Opened — "
                    f"fs={self.actual_sample_rate/1e6:.3f} MHz  "
                    f"fc={self.actual_center_freq/1e6:.6f} MHz")

    def _mark_closed(self) -> None:
        self._is_open = False
        if self._open_time is not None:
            elapsed = time.monotonic() - self._open_time
            logger.info(f"[{self.source_name}] Closed after {elapsed:.1f}s  "
                        f"({self._total_samples_read:,} samples read)")

    def _validate_samples(self, samples: np.ndarray, n_requested: int) -> np.ndarray:
        """
        Ensure output is complex64 with the expected length.
        Pads with zeros or truncates if the source delivered the wrong count.
        """
        samples = np.asarray(samples, dtype=np.complex64)
        if len(samples) < n_requested:
            pad = np.zeros(n_requested - len(samples), dtype=np.complex64)
            samples = np.concatenate([samples, pad])
            logger.debug(f"[{self.source_name}] Padded {n_requested - len(samples)} samples")
        elif len(samples) > n_requested:
            samples = samples[:n_requested]
        self._total_samples_read += n_requested
        return samples

    @staticmethod
    def power_to_dbfs(samples: np.ndarray) -> float:
        """Compute mean signal power in dBFS from a complex64 array."""
        power = float(np.mean(np.abs(samples) ** 2))
        if power <= 0:
            return -100.0
        return 10.0 * np.log10(power)

    def __repr__(self) -> str:
        status = "open" if self._is_open else "closed"
        return (f"{self.source_name}("
                f"fs={self._requested_sample_rate/1e6:.3f}MHz, "
                f"fc={self._requested_center_freq/1e6:.3f}MHz, "
                f"status={status})")
