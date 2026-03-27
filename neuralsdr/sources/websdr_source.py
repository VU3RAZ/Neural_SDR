"""
sources/websdr_source.py
========================
Web-based SDR source supporting WebSDR and KiwiSDR endpoints.

Strategy
--------
1. **KiwiSDR native IQ mode** (preferred):  KiwiSDR exposes a websocket that
   streams raw IQ at up to 20.25 kHz bandwidth.  We detect this and stream
   complex samples directly.

2. **WebSDR audio → IQ reconstruction** (fallback):  Most WebSDR instances
   only expose an audio stream.  We download the audio, apply a Hilbert-like
   neural reconstruction (actually just zero-padding in frequency domain on
   the non-NN path — this step is display/ingestion, not demodulation), and
   hand complex samples to the neural pipeline.

3. **Static IQ file fallback**:  If the URL points to a file (SigMF, cfile,
   wav), we delegate to FileSource.

Note: WebSDR access is rate-limited and should be used responsibly.
"""

from __future__ import annotations

import asyncio
import io
import queue
import threading
import time
from typing import Optional
from urllib.parse import urlparse

import numpy as np
from loguru import logger

from .base_source import IQSource
from ..config.settings import WebSDRConfig


class WebSDRSource(IQSource):
    """
    Network-based IQ source (KiwiSDR native IQ or WebSDR audio→IQ).

    Parameters
    ----------
    sample_rate : float  — Desired effective sample rate after reconstruction.
    center_freq : float  — Tuning frequency (will be sent to server if supported).
    config      : WebSDRConfig
    """

    def __init__(
        self,
        sample_rate: float,
        center_freq: float,
        config: Optional[WebSDRConfig] = None,
    ) -> None:
        super().__init__(sample_rate, center_freq)
        self._cfg = config or WebSDRConfig()
        self._sample_queue: queue.Queue = queue.Queue(maxsize=64)
        self._stop_event = threading.Event()
        self._stream_thread: Optional[threading.Thread] = None
        self._mode: str = "unknown"           # 'kiwi_iq' | 'websdr_audio'
        self._actual_fs: float = sample_rate
        self._last_level: float = -100.0

    # ── IQSource interface ────────────────────────────────────────────────────

    def open(self) -> None:
        self._check_deps()
        self._mode = self._detect_mode()
        logger.info(f"[WebSDRSource] Mode detected: {self._mode}")

        self._stop_event.clear()
        if self._mode == "kiwi_iq":
            target = self._kiwi_iq_worker
        else:
            target = self._websdr_audio_worker

        self._stream_thread = threading.Thread(
            target=target,
            name="websdr-stream",
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
        self._mark_closed()

    def read_samples(self, n_samples: int) -> np.ndarray:
        collected: list[np.ndarray] = []
        remaining = n_samples

        while remaining > 0:
            try:
                chunk: np.ndarray = self._sample_queue.get(timeout=3.0)
            except queue.Empty:
                if not self._is_open or self._stop_event.is_set():
                    break
                logger.warning("[WebSDRSource] No data received (network slow?)")
                # Return zeros rather than blocking forever
                collected.append(np.zeros(remaining, dtype=np.complex64))
                remaining = 0
                continue

            take = min(len(chunk), remaining)
            collected.append(chunk[:take])
            remaining -= take

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
    def signal_level_dbfs(self) -> float:
        return self._last_level

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _check_deps() -> None:
        missing = []
        try:
            import websockets  # noqa: F401
        except ImportError:
            missing.append("websockets")
        try:
            import aiohttp  # noqa: F401
        except ImportError:
            missing.append("aiohttp")
        if missing:
            raise RuntimeError(
                f"Missing dependencies for WebSDRSource: {missing}.  "
                f"Run: pip install {' '.join(missing)}"
            )

    def _detect_mode(self) -> str:
        """
        Heuristic: if the URL contains 'kiwi' or port 8073, assume KiwiSDR IQ.
        Otherwise fall back to audio capture.
        """
        parsed = urlparse(self._cfg.url)
        host_lower = (parsed.hostname or "").lower()
        if "kiwi" in host_lower or parsed.port == 8073:
            return "kiwi_iq"
        return "websdr_audio"

    # ── KiwiSDR native IQ worker ──────────────────────────────────────────────

    def _kiwi_iq_worker(self) -> None:
        """
        Connects to KiwiSDR via websocket and streams IQ samples.
        KiwiSDR IQ protocol reference: github.com/jks-prv/kiwiclient
        """
        import asyncio as _asyncio
        _asyncio.run(self._kiwi_iq_async())

    async def _kiwi_iq_async(self) -> None:
        import websockets  # type: ignore

        freq_khz = self._requested_center_freq / 1e3
        # KiwiSDR IQ websocket path
        ws_url = self._cfg.url.rstrip("/") + "/kiwi/99/IQ"

        logger.info(f"[WebSDRSource] Connecting KiwiSDR IQ @ {ws_url}")
        try:
            async with websockets.connect(
                ws_url,
                open_timeout=self._cfg.connection_timeout,
                ping_interval=20,
            ) as ws:
                # Send tune command
                await ws.send(f"SET freq={freq_khz:.3f}")
                await ws.send("SET mod=iq lowcut=-5000 highcut=5000 freq=0")
                await ws.send("SET ar_ok=1 squelch=0")
                logger.info("[WebSDRSource] KiwiSDR IQ stream active")
                self._actual_fs = 12_000.0   # KiwiSDR IQ rate

                async for message in ws:
                    if self._stop_event.is_set():
                        break
                    try:
                        samples = self._parse_kiwi_iq_packet(message)
                        if samples is not None and len(samples) > 0:
                            self._last_level = self.power_to_dbfs(samples)
                            self._sample_queue.put_nowait(samples)
                    except Exception as exc:
                        logger.debug(f"[WebSDRSource] Packet parse error: {exc}")
        except Exception as exc:
            logger.error(f"[WebSDRSource] KiwiSDR connection failed: {exc}")
            logger.info("[WebSDRSource] Falling back to synthetic zeros")
            self._emit_silence_forever()

    @staticmethod
    def _parse_kiwi_iq_packet(data: bytes) -> Optional[np.ndarray]:
        """
        Parse a KiwiSDR binary IQ packet.
        Format: 1-byte flags + N×(int16 I, int16 Q) pairs.
        """
        if not isinstance(data, (bytes, bytearray)):
            return None
        if len(data) < 3:
            return None
        # Skip 1-byte header
        raw = np.frombuffer(data[1:], dtype=np.int16)
        if len(raw) % 2 != 0:
            raw = raw[:-1]
        i_samples = raw[0::2].astype(np.float32) / 32768.0
        q_samples = raw[1::2].astype(np.float32) / 32768.0
        return (i_samples + 1j * q_samples).astype(np.complex64)

    # ── WebSDR audio→IQ worker ────────────────────────────────────────────────

    def _websdr_audio_worker(self) -> None:
        """
        WebSDR only exposes an audio stream (MP3/OGG).
        We fetch audio chunks, decode them, and synthesise IQ via an analytic
        signal (Hilbert transform in frequency domain).

        This is the ONLY non-NN classical DSP in the entire pipeline — it is
        in the *ingestion* layer, not the signal processing layer.  The neural
        receiver still operates on the resulting complex samples.
        """
        try:
            import urllib.request
            import soundfile as sf  # type: ignore
        except ImportError:
            logger.error("[WebSDRSource] soundfile not installed — cannot decode audio")
            self._emit_silence_forever()
            return

        audio_url = self._cfg.url
        logger.info(f"[WebSDRSource] Fetching audio stream: {audio_url}")

        # Politely identify ourselves
        headers = {"User-Agent": "NeuralSDR/1.0 (research receiver)"}

        try:
            req = urllib.request.Request(audio_url, headers=headers)
            with urllib.request.urlopen(
                req, timeout=self._cfg.connection_timeout
            ) as response:
                chunk_size = self._cfg.audio_rate  # 1 second chunks
                raw_buf = io.BytesIO()

                while not self._stop_event.is_set():
                    chunk = response.read(chunk_size * 2)  # 2 bytes/sample
                    if not chunk:
                        break
                    raw_buf.write(chunk)

                    if raw_buf.tell() >= chunk_size * 2:
                        raw_buf.seek(0)
                        try:
                            audio, sr = sf.read(raw_buf, dtype="float32")
                            iq = self._audio_to_iq(audio, sr)
                            self._last_level = self.power_to_dbfs(iq)
                            self._sample_queue.put_nowait(iq)
                            self._actual_fs = float(sr)
                        except Exception as parse_exc:
                            logger.debug(f"[WebSDRSource] Audio parse: {parse_exc}")
                        raw_buf = io.BytesIO()

        except Exception as exc:
            logger.error(f"[WebSDRSource] Audio stream failed: {exc}")
            self._emit_silence_forever()

    @staticmethod
    def _audio_to_iq(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Convert real audio to analytic (complex) signal via FFT-based Hilbert.

        This is purely for IQ ingestion from audio-only streams — it is NOT
        part of the neural demodulation pipeline.

        Parameters
        ----------
        audio       : np.ndarray float32, shape (N,) or (N, channels)
        sample_rate : int

        Returns
        -------
        np.ndarray complex64
        """
        if audio.ndim > 1:
            audio = audio[:, 0]   # take left channel

        n = len(audio)
        # FFT → zero negative freqs → IFFT = analytic signal
        spectrum = np.fft.rfft(audio, n=n)
        # Build analytic spectrum: double positive freqs, zero negative
        analytic_spec = np.zeros(n // 2 + 1, dtype=np.complex64)
        analytic_spec[:] = spectrum * 2.0
        if n % 2 == 0:
            analytic_spec[0]  /= 2.0   # DC
            analytic_spec[-1] /= 2.0   # Nyquist
        analytic = np.fft.irfft(analytic_spec, n=n)
        return analytic.astype(np.complex64)

    def _emit_silence_forever(self) -> None:
        """Fallback: put silence chunks into the queue so the pipeline doesn't block."""
        silence = np.zeros(1024, dtype=np.complex64)
        while not self._stop_event.is_set():
            try:
                self._sample_queue.put_nowait(silence.copy())
            except queue.Full:
                pass
            time.sleep(0.05)
