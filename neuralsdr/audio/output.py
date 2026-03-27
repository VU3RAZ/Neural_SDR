"""
audio/output.py
===============
Real-time PCM audio output for NeuralSDR.

Handles:
  - Continuous playback via sounddevice callback (low-latency ring buffer)
  - Volume control and soft muting
  - Squelch gating (silence output when neural squelch is closed)
  - Optional WAV recording of demodulated audio
  - Resampling: model audio rate → system audio rate

Design
------
The neural analog head produces PCM at a rate that depends on the
model's internal sample rate and upsample factor.  sounddevice plays
back at the system's native rate (usually 44100 or 48000 Hz).  We use
a simple linear interpolation resampler to bridge the gap — this is
only for *playback* and is not in the signal-processing path.

Thread safety
-------------
The sounddevice callback runs in a high-priority audio thread.
We use a lock-free ring buffer (numpy circular array + atomic indices
approximated with threading.Lock) so the main thread can push audio
without blocking the callback.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Lock-aware ring buffer
# ─────────────────────────────────────────────────────────────────────────────

class AudioRingBuffer:
    """
    Thread-safe ring buffer for streaming PCM audio.

    Producer (main thread) pushes float32 mono samples.
    Consumer (sounddevice callback) pops float32 mono samples.

    Parameters
    ----------
    capacity : int — Ring buffer capacity in samples.
    """

    def __init__(self, capacity: int = 88200) -> None:  # 2 s at 44100 Hz
        self._buf  = np.zeros(capacity, dtype=np.float32)
        self._cap  = capacity
        self._head = 0   # write position
        self._tail = 0   # read position
        self._lock = threading.Lock()

    @property
    def available(self) -> int:
        """Samples available for reading."""
        with self._lock:
            return (self._head - self._tail) % self._cap

    @property
    def free(self) -> int:
        """Free slots for writing."""
        return self._cap - self.available - 1

    def push(self, samples: np.ndarray) -> int:
        """
        Write samples into the buffer.  Drops oldest data if full.

        Returns
        -------
        int  — Number of samples actually written.
        """
        n = len(samples)
        with self._lock:
            if n > self._cap - 1:
                samples = samples[-self._cap + 1:]
                n = len(samples)
            # Overwrite old data if nearly full (prefer new audio)
            space = (self._cap - 1 - (self._head - self._tail) % self._cap)
            if n > space:
                # Advance tail to make room
                discard = n - space
                self._tail = (self._tail + discard) % self._cap

            # Write in up to two segments (wrap-around)
            end = self._head + n
            if end <= self._cap:
                self._buf[self._head:end] = samples
            else:
                first = self._cap - self._head
                self._buf[self._head:] = samples[:first]
                self._buf[:n - first]  = samples[first:]
            self._head = (self._head + n) % self._cap
            return n

    def pop(self, n: int) -> np.ndarray:
        """
        Read up to n samples.  Returns zeros if buffer is empty.

        Parameters
        ----------
        n : int

        Returns
        -------
        np.ndarray  float32  shape (n,)
        """
        with self._lock:
            avail = (self._head - self._tail) % self._cap
            read  = min(n, avail)
            out   = np.zeros(n, dtype=np.float32)
            if read == 0:
                return out

            end = self._tail + read
            if end <= self._cap:
                out[:read] = self._buf[self._tail:end]
            else:
                first = self._cap - self._tail
                out[:first]      = self._buf[self._tail:]
                out[first:read]  = self._buf[:read - first]
            self._tail = (self._tail + read) % self._cap
            return out

    def clear(self) -> None:
        with self._lock:
            self._head = 0
            self._tail = 0


# ─────────────────────────────────────────────────────────────────────────────
# Resampler (linear interpolation, playback only)
# ─────────────────────────────────────────────────────────────────────────────

def resample_linear(
    audio:      np.ndarray,
    in_rate:    float,
    out_rate:   float,
) -> np.ndarray:
    """
    Simple linear interpolation resampler for playback rate matching.

    NOT used in the neural signal path — only for driving sounddevice.

    Parameters
    ----------
    audio    : np.ndarray  float32  shape (N,)
    in_rate  : float — Input sample rate.
    out_rate : float — Output sample rate.

    Returns
    -------
    np.ndarray  float32
    """
    if abs(in_rate - out_rate) < 1.0:
        return audio
    ratio    = out_rate / in_rate
    n_out    = max(1, int(len(audio) * ratio))
    x_in     = np.arange(len(audio), dtype=np.float64)
    x_out    = np.linspace(0, len(audio) - 1, n_out)
    return np.interp(x_out, x_in, audio.astype(np.float64)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Audio output manager
# ─────────────────────────────────────────────────────────────────────────────

class AudioOutput:
    """
    Real-time PCM audio playback and optional WAV recording.

    Parameters
    ----------
    output_rate   : int   — Playback sample rate (Hz).  Must match sounddevice.
    model_rate    : int   — Rate of audio coming from the neural analog head.
    buffer_size   : int   — sounddevice buffer size (samples per callback).
    volume        : float — Initial volume (0.0 – 1.0).
    squelch_threshold: float — Min presence_prob to open squelch (0 = always open).
    recording_path: str or None — Write demodulated audio to this WAV path.
    """

    def __init__(
        self,
        output_rate:       int   = 44100,
        model_rate:        int   = 44100,
        buffer_size:       int   = 2048,
        volume:            float = 0.8,
        squelch_threshold: float = 0.3,
        recording_path:    Optional[str] = None,
    ) -> None:
        self.output_rate       = output_rate
        self.model_rate        = model_rate
        self.buffer_size       = buffer_size
        self.volume            = float(np.clip(volume, 0.0, 1.0))
        self.squelch_threshold = squelch_threshold
        self.recording_path    = recording_path

        self._ring   = AudioRingBuffer(capacity=output_rate * 4)  # 4 s buffer
        self._stream = None     # sounddevice OutputStream
        self._wav_writer = None
        self._is_open    = False
        self._muted      = False
        self._squelch_open = True

        # Statistics
        self._samples_played   = 0
        self._underrun_count   = 0

    # ── Public interface ──────────────────────────────────────────────────────

    def open(self) -> None:
        """Open the sounddevice stream and optional WAV recorder."""
        self._open_stream()
        if self.recording_path:
            self._open_recorder()
        self._is_open = True
        logger.info(
            f"[AudioOutput] Opened — "
            f"rate={self.output_rate} Hz  "
            f"buf={self.buffer_size}  "
            f"vol={self.volume:.2f}  "
            f"squelch_threshold={self.squelch_threshold:.2f}"
        )

    def close(self) -> None:
        if not self._is_open:
            return
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._wav_writer is not None:
            try:
                self._wav_writer.close()
            except Exception:
                pass
            self._wav_writer = None
        self._is_open = False
        logger.info(
            f"[AudioOutput] Closed — "
            f"{self._samples_played:,} samples played  "
            f"{self._underrun_count} underruns"
        )

    def push_audio(
        self,
        pcm:            np.ndarray,
        presence_prob:  float = 1.0,
    ) -> None:
        """
        Push demodulated PCM audio into the playback buffer.

        Parameters
        ----------
        pcm          : np.ndarray  float32  shape (N,) or (1, N)
        presence_prob: float — Neural squelch probability (0–1).
                       If < squelch_threshold, audio is silenced.
        """
        if not self._is_open:
            return

        # Flatten
        pcm = np.asarray(pcm, dtype=np.float32).ravel()

        # Squelch gate
        self._squelch_open = presence_prob >= self.squelch_threshold
        if not self._squelch_open or self._muted:
            pcm = np.zeros_like(pcm)

        # Apply volume
        pcm = pcm * self.volume

        # Clip to prevent speaker damage
        pcm = np.clip(pcm, -1.0, 1.0)

        # Resample to playback rate
        if abs(self.model_rate - self.output_rate) > 1:
            pcm = resample_linear(pcm, self.model_rate, self.output_rate)

        # Push to ring buffer
        self._ring.push(pcm)

        # Write to WAV recorder if active
        if self._wav_writer is not None and self._squelch_open:
            try:
                self._wav_writer.write(pcm)
            except Exception as exc:
                logger.debug(f"[AudioOutput] WAV write error: {exc}")

    def set_volume(self, volume: float) -> None:
        self.volume = float(np.clip(volume, 0.0, 1.0))

    def set_squelch_threshold(self, threshold: float) -> None:
        self.squelch_threshold = float(np.clip(threshold, 0.0, 1.0))

    def mute(self) -> None:
        self._muted = True

    def unmute(self) -> None:
        self._muted = False

    def clear_buffer(self) -> None:
        self._ring.clear()

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def squelch_open(self) -> bool:
        return self._squelch_open

    @property
    def buffer_fill_pct(self) -> float:
        return 100.0 * self._ring.available / self._ring._cap

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _open_stream(self) -> None:
        """Open sounddevice OutputStream."""
        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            logger.warning(
                "[AudioOutput] sounddevice not installed — audio output disabled.  "
                "Run: pip install sounddevice"
            )
            self._stream = None
            return

        def _callback(outdata: np.ndarray, frames: int, time_info, status):
            if status.output_underflow:
                self._underrun_count += 1
            samples = self._ring.pop(frames)
            outdata[:, 0] = samples
            self._samples_played += frames

        try:
            self._stream = sd.OutputStream(
                samplerate=self.output_rate,
                channels=1,
                dtype="float32",
                blocksize=self.buffer_size,
                callback=_callback,
            )
            self._stream.start()
        except Exception as exc:
            logger.warning(f"[AudioOutput] Could not open audio device: {exc}")
            self._stream = None

    def _open_recorder(self) -> None:
        """Open soundfile WAV writer for recording."""
        try:
            import soundfile as sf  # type: ignore
        except ImportError:
            logger.warning("[AudioOutput] soundfile not installed — recording disabled")
            return

        os.makedirs(os.path.dirname(os.path.abspath(self.recording_path)), exist_ok=True)
        try:
            self._wav_writer = sf.SoundFile(
                self.recording_path,
                mode="w",
                samplerate=self.output_rate,
                channels=1,
                format="WAV",
                subtype="FLOAT",
            )
            logger.info(f"[AudioOutput] Recording → {self.recording_path!r}")
        except Exception as exc:
            logger.warning(f"[AudioOutput] Could not open WAV recorder: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Null audio output (for headless / test mode)
# ─────────────────────────────────────────────────────────────────────────────

class NullAudioOutput:
    """
    Drop-in replacement for AudioOutput that discards all audio.
    Used in headless mode, testing, or when sounddevice is unavailable.
    """

    def __init__(self, *args, **kwargs):
        self.volume            = kwargs.get("volume", 0.8)
        self.squelch_threshold = kwargs.get("squelch_threshold", 0.3)
        self._is_open          = False
        self._squelch_open     = True

    def open(self)                                          : self._is_open = True
    def close(self)                                         : self._is_open = False
    def push_audio(self, pcm, presence_prob=1.0)            : pass
    def set_volume(self, v)                                 : self.volume = v
    def set_squelch_threshold(self, t)                      : self.squelch_threshold = t
    def mute(self)                                          : pass
    def unmute(self)                                        : pass
    def clear_buffer(self)                                  : pass

    @property
    def is_open(self)       : return self._is_open
    @property
    def squelch_open(self)  : return self._squelch_open
    @property
    def buffer_fill_pct(self): return 0.0

    def __enter__(self): self.open(); return self
    def __exit__(self, *a): self.close()


def make_audio_output(cfg, headless: bool = False) -> AudioOutput:
    """
    Factory: return the correct AudioOutput for the current environment.

    Parameters
    ----------
    cfg      : AudioConfig
    headless : bool — If True, always returns NullAudioOutput.

    Returns
    -------
    AudioOutput or NullAudioOutput
    """
    if headless:
        return NullAudioOutput(
            volume=cfg.volume,
            squelch_threshold=cfg.squelch_threshold,
        )

    return AudioOutput(
        output_rate       = cfg.output_sample_rate,
        model_rate        = cfg.output_sample_rate,
        buffer_size       = cfg.buffer_size,
        volume            = cfg.volume,
        squelch_threshold = cfg.squelch_threshold,
        recording_path    = cfg.recording_path,
    )
