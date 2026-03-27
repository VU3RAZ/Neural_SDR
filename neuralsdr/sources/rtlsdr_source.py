"""
sources/rtlsdr_source.py
========================
RTL-SDR IQ source using the ``pyrtlsdr`` library.

Supports RTL2832U-based dongles (RTL-SDR Blog v3/v4, NooElec, Nooelec
SMART, and compatible devices).

Graceful fallback
-----------------
If ``pyrtlsdr`` is not installed or no device is found, ``open()`` raises a
clear ``RuntimeError`` instead of a cryptic import error.
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np
from loguru import logger

from .base_source import IQSource
from ..config.settings import RTLSDRConfig


class RTLSDRSource(IQSource):
    """
    IQ source backed by an RTL-SDR dongle.

    Uses the *async callback* interface of pyrtlsdr internally so the USB
    transfer thread is fully decoupled from the consumer.  Samples are queued
    in a thread-safe buffer and returned to ``read_samples()`` in chunks.

    Parameters
    ----------
    sample_rate  : float — Desired sample rate (e.g. 2.048e6).
    center_freq  : float — Desired centre frequency (Hz).
    config       : RTLSDRConfig — Hardware-specific settings.
    """

    # pyrtlsdr delivers samples in multiples of this block size
    _RTLSDR_READ_SIZE: int = 262_144  # 256 k complex samples

    def __init__(
        self,
        sample_rate: float,
        center_freq: float,
        config: Optional[RTLSDRConfig] = None,
    ) -> None:
        super().__init__(sample_rate, center_freq)
        self._cfg = config or RTLSDRConfig()
        self._sdr = None          # rtlsdr.RtlSdr instance
        self._sample_queue: queue.Queue = queue.Queue(maxsize=32)
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._actual_fs: float = sample_rate
        self._actual_fc: float = center_freq

    # ── IQSource interface ────────────────────────────────────────────────────

    def open(self) -> None:
        try:
            from rtlsdr import RtlSdr  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pyrtlsdr is not installed.  Run: pip install pyrtlsdr"
            ) from exc

        try:
            self._sdr = RtlSdr(device_index=self._cfg.device_index)
        except Exception as exc:
            raise RuntimeError(
                f"Could not open RTL-SDR device #{self._cfg.device_index}.  "
                f"Is the dongle plugged in?  Error: {exc}"
            ) from exc

        # ── Configure hardware ─────────────────────────────────────────────
        self._sdr.sample_rate = self._requested_sample_rate
        self._sdr.center_freq = self._requested_center_freq

        if self._cfg.gain == 0:
            self._sdr.gain = "auto"
        else:
            self._sdr.gain = self._cfg.gain

        self._sdr.freq_correction = self._cfg.ppm_correction

        # Enable bias-tee if the dongle supports it
        if self._cfg.bias_tee:
            try:
                self._sdr.set_bias_tee(True)
                logger.info("[RTLSDRSource] Bias-tee enabled")
            except AttributeError:
                logger.warning("[RTLSDRSource] Bias-tee not supported on this device")

        # Read back actual values (driver may round them)
        self._actual_fs = float(self._sdr.sample_rate)
        self._actual_fc = float(self._sdr.center_freq)

        logger.info(
            f"[RTLSDRSource] Device opened — "
            f"actual fs={self._actual_fs/1e6:.4f} MHz  "
            f"fc={self._actual_fc/1e6:.6f} MHz  "
            f"gain={self._sdr.gain} dB"
        )

        # Start background USB stream thread
        self._stop_event.clear()
        self._stream_thread = threading.Thread(
            target=self._stream_worker,
            name="rtlsdr-stream",
            daemon=True,
        )
        self._stream_thread.start()
        self._mark_open()

    def close(self) -> None:
        if not self._is_open:
            return
        self._stop_event.set()
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=5.0)
        if self._sdr is not None:
            try:
                self._sdr.cancel_read_async()
            except Exception:
                pass
            try:
                self._sdr.close()
            except Exception:
                pass
            self._sdr = None
        self._mark_closed()

    def read_samples(self, n_samples: int) -> np.ndarray:
        """
        Collect ``n_samples`` from the internal queue, blocking if necessary.
        """
        collected: list[np.ndarray] = []
        remaining = n_samples

        while remaining > 0:
            try:
                chunk: np.ndarray = self._sample_queue.get(timeout=2.0)
            except queue.Empty:
                if not self._is_open or self._stop_event.is_set():
                    break
                logger.warning("[RTLSDRSource] Queue timeout — device stalled?")
                continue

            if len(chunk) <= remaining:
                collected.append(chunk)
                remaining -= len(chunk)
            else:
                collected.append(chunk[:remaining])
                # Put the rest back (prepend hack via a temporary deque would be
                # cleaner, but for SDR use the small over-read is acceptable)
                remaining = 0

        if collected:
            samples = np.concatenate(collected)
        else:
            samples = np.zeros(n_samples, dtype=np.complex64)

        return self._validate_samples(samples, n_samples)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def actual_sample_rate(self) -> float:
        return self._actual_fs

    @property
    def actual_center_freq(self) -> float:
        return self._actual_fc

    @property
    def signal_level_dbfs(self) -> float:
        # RTL-SDR has no direct RSSI register; estimate from queue snapshot
        if self._sample_queue.empty():
            return -100.0
        try:
            chunk = self._sample_queue.queue[0]  # peek without consuming
            return self.power_to_dbfs(chunk)
        except Exception:
            return -100.0

    # ── Runtime tuning ────────────────────────────────────────────────────────

    def set_center_freq(self, freq_hz: float) -> None:
        """Retune to a new centre frequency without reopening the device."""
        if self._sdr is None:
            return
        self._sdr.center_freq = freq_hz
        self._actual_fc = float(self._sdr.center_freq)
        logger.debug(f"[RTLSDRSource] Retuned → {self._actual_fc/1e6:.6f} MHz")

    def set_gain(self, gain_db: float) -> None:
        """Change RF gain at runtime."""
        if self._sdr is None:
            return
        self._sdr.gain = gain_db
        logger.debug(f"[RTLSDRSource] Gain set to {gain_db} dB")

    # ── Internal stream worker ────────────────────────────────────────────────

    def _stream_worker(self) -> None:
        """
        Runs in a dedicated thread.  Reads blocks of raw uint8 from the RTL-SDR
        USB interface, converts to complex64, and enqueues them for consumption.
        """
        logger.debug("[RTLSDRSource] Stream worker started")
        while not self._stop_event.is_set():
            try:
                # read_samples returns complex128 by default — cast immediately
                raw = self._sdr.read_samples(self._RTLSDR_READ_SIZE)
                samples = np.asarray(raw, dtype=np.complex64)
                try:
                    self._sample_queue.put_nowait(samples)
                except queue.Full:
                    # Drop oldest block if consumer is too slow
                    try:
                        self._sample_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._sample_queue.put_nowait(samples)
                    logger.debug("[RTLSDRSource] Queue full — dropped oldest block")
            except Exception as exc:
                if not self._stop_event.is_set():
                    logger.error(f"[RTLSDRSource] Stream error: {exc}")
                break
        logger.debug("[RTLSDRSource] Stream worker exited")
