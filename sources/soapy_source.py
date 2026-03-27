"""
sources/soapy_source.py
=======================
SoapySDR-backed IQ source for HackRF, Airspy, LimeSDR, PlutoSDR, etc.

SoapySDR is a vendor-neutral hardware abstraction layer.  Any radio that
has a SoapySDR module installed (system package or pip) will work here
without any code changes — just pass the correct ``driver`` string in
``SoapyConfig``.

System install (required before pip install):
  Ubuntu:  sudo apt install libsoapysdr-dev python3-soapysdr
  macOS:   brew install soapysdr
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np
from loguru import logger

from .base_source import IQSource
from ..config.settings import SoapyConfig


class SoapySDRSource(IQSource):
    """
    Universal SDR source using SoapySDR.

    Supports HackRF One, Airspy R2 / Mini, LimeSDR, PlutoSDR, RTL-SDR
    (via driver='rtlsdr'), and any device with a SoapySDR module.

    Parameters
    ----------
    sample_rate  : float — Desired sample rate.
    center_freq  : float — Desired centre frequency.
    config       : SoapyConfig — Driver name, gain, antenna.
    """

    _SOAPY_CHUNK: int = 131_072   # Samples per readStream() call

    def __init__(
        self,
        sample_rate: float,
        center_freq: float,
        config: Optional[SoapyConfig] = None,
    ) -> None:
        super().__init__(sample_rate, center_freq)
        self._cfg = config or SoapyConfig()
        self._device = None      # SoapySDR.Device
        self._stream = None      # RX stream handle
        self._sample_queue: queue.Queue = queue.Queue(maxsize=32)
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._actual_fs: float = sample_rate
        self._actual_fc: float = center_freq

    # ── IQSource interface ────────────────────────────────────────────────────

    def open(self) -> None:
        try:
            import SoapySDR  # type: ignore
            from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "SoapySDR Python bindings not found.\n"
                "Install system package first:\n"
                "  Ubuntu: sudo apt install python3-soapysdr\n"
                "  macOS:  brew install soapysdr && pip install pysoapysdr"
            ) from exc

        # Build driver args string
        args = f"driver={self._cfg.driver}"
        if self._cfg.serial:
            args += f",serial={self._cfg.serial}"

        try:
            self._device = SoapySDR.Device(args)
        except Exception as exc:
            raise RuntimeError(
                f"SoapySDR: could not open device with args='{args}'.  "
                f"Is the hardware connected?  Error: {exc}"
            ) from exc

        # ── Configure ──────────────────────────────────────────────────────
        SOAPY_SDR_RX = 1  # channel direction constant

        self._device.setSampleRate(SOAPY_SDR_RX, 0, self._requested_sample_rate)
        self._device.setFrequency(SOAPY_SDR_RX, 0, self._requested_center_freq)
        self._device.setAntenna(SOAPY_SDR_RX, 0, self._cfg.antenna)

        # Set gain — some devices use named gain elements
        try:
            self._device.setGainMode(SOAPY_SDR_RX, 0, False)  # manual gain
            self._device.setGain(SOAPY_SDR_RX, 0, self._cfg.gain_db)
        except Exception as exc:
            logger.warning(f"[SoapySDRSource] Could not set gain: {exc}")

        self._actual_fs = self._device.getSampleRate(SOAPY_SDR_RX, 0)
        self._actual_fc = self._device.getFrequency(SOAPY_SDR_RX, 0)

        logger.info(
            f"[SoapySDRSource] Device='{self._cfg.driver}'  "
            f"fs={self._actual_fs/1e6:.4f} MHz  "
            f"fc={self._actual_fc/1e6:.6f} MHz"
        )

        # ── Open stream ────────────────────────────────────────────────────
        SOAPY_SDR_CF32 = "CF32"
        self._stream = self._device.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        self._device.activateStream(self._stream)

        self._stop_event.clear()
        self._stream_thread = threading.Thread(
            target=self._stream_worker,
            name="soapy-stream",
            daemon=True,
            args=(SoapySDR,),
        )
        self._stream_thread.start()
        self._mark_open()

    def close(self) -> None:
        if not self._is_open:
            return
        self._stop_event.set()
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=5.0)
        if self._stream is not None and self._device is not None:
            try:
                self._device.deactivateStream(self._stream)
                self._device.closeStream(self._stream)
            except Exception:
                pass
            self._stream = None
        if self._device is not None:
            try:
                import SoapySDR  # type: ignore
                SoapySDR.Device.unmake(self._device)
            except Exception:
                pass
            self._device = None
        self._mark_closed()

    def read_samples(self, n_samples: int) -> np.ndarray:
        collected: list[np.ndarray] = []
        remaining = n_samples

        while remaining > 0:
            try:
                chunk: np.ndarray = self._sample_queue.get(timeout=2.0)
            except queue.Empty:
                if not self._is_open or self._stop_event.is_set():
                    break
                logger.warning("[SoapySDRSource] Read timeout")
                continue

            if len(chunk) <= remaining:
                collected.append(chunk)
                remaining -= len(chunk)
            else:
                collected.append(chunk[:remaining])
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

    # ── Runtime tuning ────────────────────────────────────────────────────────

    def set_center_freq(self, freq_hz: float) -> None:
        if self._device is None:
            return
        SOAPY_SDR_RX = 1
        self._device.setFrequency(SOAPY_SDR_RX, 0, freq_hz)
        self._actual_fc = self._device.getFrequency(SOAPY_SDR_RX, 0)
        logger.debug(f"[SoapySDRSource] Retuned → {self._actual_fc/1e6:.6f} MHz")

    def set_gain(self, gain_db: float) -> None:
        if self._device is None:
            return
        SOAPY_SDR_RX = 1
        self._device.setGain(SOAPY_SDR_RX, 0, gain_db)
        logger.debug(f"[SoapySDRSource] Gain → {gain_db} dB")

    # ── Internal stream worker ────────────────────────────────────────────────

    def _stream_worker(self, SoapySDR) -> None:
        logger.debug("[SoapySDRSource] Stream worker started")
        SOAPY_SDR_RX = 1
        buff = np.zeros(self._SOAPY_CHUNK, dtype=np.complex64)

        while not self._stop_event.is_set():
            try:
                sr = self._device.readStream(
                    self._stream, [buff], self._SOAPY_CHUNK, timeoutUs=1_000_000
                )
                if sr.ret < 0:
                    if not self._stop_event.is_set():
                        logger.warning(f"[SoapySDRSource] readStream error: {sr.ret}")
                    continue

                samples = buff[:sr.ret].copy()
                try:
                    self._sample_queue.put_nowait(samples)
                except queue.Full:
                    try:
                        self._sample_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._sample_queue.put_nowait(samples)

            except Exception as exc:
                if not self._stop_event.is_set():
                    logger.error(f"[SoapySDRSource] Stream error: {exc}")
                break

        logger.debug("[SoapySDRSource] Stream worker exited")
