"""
config/settings.py
==================
Central configuration for NeuralSDR.

All runtime parameters live here as typed, validated Pydantic dataclasses.
Other modules import from this file — never hardcode magic numbers elsewhere.
"""

from __future__ import annotations

import os
from enum import Enum, auto
from typing import Optional, List, Tuple

import torch
from pydantic import BaseModel, Field, field_validator, model_validator
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class SourceType(str, Enum):
    """IQ acquisition source."""
    RTL_SDR    = "rtlsdr"       # RTL-SDR USB dongle via pyrtlsdr
    SOAPY      = "soapy"        # HackRF / Airspy / LimeSDR via SoapySDR
    WEBSDR     = "websdr"       # WebSDR / KiwiSDR network stream
    FILE       = "file"         # SigMF / complex WAV / raw .cfile
    SYNTHETIC  = "synthetic"    # Built-in synthetic IQ generator (testing)


class ModulationMode(str, Enum):
    """
    Supported modulation schemes.
    The neural receiver uses these as conditioning labels.
    Integer indices are used for one-hot embeddings — order matters.
    """
    # ── Analog ────────────────────────────────────────────────────────────────
    AM      = "AM"          # Amplitude Modulation
    FM_NB   = "FM_NB"       # FM Narrowband (≤16 kHz deviation, voice)
    FM_WB   = "FM_WB"       # FM Wideband (75 kHz deviation, broadcast)
    USB     = "USB"         # Upper Sideband SSB
    LSB     = "LSB"         # Lower Sideband SSB
    CW      = "CW"          # Continuous Wave (Morse)
    DSB     = "DSB"         # Double Sideband (suppressed carrier)

    # ── Digital ───────────────────────────────────────────────────────────────
    BPSK    = "BPSK"        # Binary Phase Shift Keying
    QPSK    = "QPSK"        # Quadrature PSK
    PSK8    = "8PSK"        # 8-ary PSK
    QAM16   = "16QAM"       # 16-QAM
    QAM64   = "64QAM"       # 64-QAM
    GFSK    = "GFSK"        # Gaussian FSK (Bluetooth-like)
    CPFSK   = "CPFSK"       # Continuous Phase FSK
    OFDM    = "OFDM"        # Basic OFDM (extensible)

    @classmethod
    def analog_modes(cls) -> List["ModulationMode"]:
        return [cls.AM, cls.FM_NB, cls.FM_WB, cls.USB, cls.LSB, cls.CW, cls.DSB]

    @classmethod
    def digital_modes(cls) -> List["ModulationMode"]:
        return [cls.BPSK, cls.QPSK, cls.PSK8, cls.QAM16, cls.QAM64,
                cls.GFSK, cls.CPFSK, cls.OFDM]

    @property
    def is_analog(self) -> bool:
        return self in ModulationMode.analog_modes()

    @property
    def is_digital(self) -> bool:
        return self in ModulationMode.digital_modes()

    @property
    def index(self) -> int:
        """Integer index for one-hot embedding — stable across runs."""
        return list(ModulationMode).index(self)

    @property
    def num_constellation_points(self) -> int:
        """Number of constellation points (digital modes only)."""
        _map = {
            ModulationMode.BPSK:  2,
            ModulationMode.QPSK:  4,
            ModulationMode.PSK8:  8,
            ModulationMode.QAM16: 16,
            ModulationMode.QAM64: 64,
            ModulationMode.GFSK:  2,
            ModulationMode.CPFSK: 2,
            ModulationMode.OFDM:  4,   # QPSK sub-carriers by default
        }
        return _map.get(self, 2)


NUM_MODULATIONS: int = len(ModulationMode)


# ─────────────────────────────────────────────────────────────────────────────
# Device / Compute Config
# ─────────────────────────────────────────────────────────────────────────────

class DeviceConfig(BaseModel):
    """PyTorch compute device selection."""
    model_config = {"arbitrary_types_allowed": True}

    device_str: str = Field(
        default="auto",
        description="'auto' | 'cpu' | 'cuda' | 'cuda:N' | 'mps'",
    )
    use_compile: bool = Field(
        default=True,
        description="Apply torch.compile() for inference speedup (requires PyTorch 2.x).",
    )
    dtype_str: str = Field(
        default="float32",
        description="Inference dtype: 'float32' | 'float16' | 'bfloat16'.",
    )

    @property
    def device(self) -> torch.device:
        if self.device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device_str)

    @property
    def dtype(self) -> torch.dtype:
        return {
            "float32":  torch.float32,
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.dtype_str]

    def log_info(self) -> None:
        d = self.device
        logger.info(f"[DeviceConfig] Using device: {d}")
        if d.type == "cuda":
            logger.info(f"  GPU: {torch.cuda.get_device_name(d)}")
            mem = torch.cuda.get_device_properties(d).total_memory / 1e9
            logger.info(f"  VRAM: {mem:.1f} GB")
        logger.info(f"  dtype: {self.dtype_str}  |  torch.compile: {self.use_compile}")


# ─────────────────────────────────────────────────────────────────────────────
# Source Configs
# ─────────────────────────────────────────────────────────────────────────────

class RTLSDRConfig(BaseModel):
    device_index: int = Field(default=0, ge=0, description="USB device index.")
    gain: float = Field(default=30.0, ge=0.0, le=60.0, description="RF gain in dB.")
    ppm_correction: int = Field(default=0, description="Frequency correction in PPM.")
    bias_tee: bool = Field(default=False, description="Enable bias-tee (RTL-SDR Blog v3+).")


class SoapyConfig(BaseModel):
    driver: str = Field(default="hackrf", description="SoapySDR driver name.")
    serial: Optional[str] = Field(default=None, description="Device serial (multi-device).")
    gain_db: float = Field(default=30.0, ge=0.0)
    antenna: str = Field(default="RX", description="Antenna port name.")


class WebSDRConfig(BaseModel):
    url: str = Field(
        default="http://websdr.ewi.utwente.nl:8901",
        description="WebSDR or KiwiSDR base URL.",
    )
    password: Optional[str] = Field(default=None)
    audio_rate: int = Field(default=12000, description="Audio sample rate from WebSDR.")
    connection_timeout: float = Field(default=10.0)


class FileSourceConfig(BaseModel):
    path: str = Field(default="", description="Path to IQ recording file.")
    loop: bool = Field(default=True, description="Loop file playback.")
    playback_rate: float = Field(
        default=1.0, gt=0.0,
        description="Playback speed multiplier (1.0 = real-time).",
    )

    @field_validator("path")
    @classmethod
    def path_must_exist_if_set(cls, v: str) -> str:
        if v and not os.path.exists(v):
            raise ValueError(f"IQ file not found: {v!r}")
        return v


class SyntheticConfig(BaseModel):
    noise_floor_db: float = Field(default=-20.0, description="AWGN noise floor (dBFS).")
    snr_db: float = Field(default=15.0, description="Signal SNR above noise floor.")
    freq_offset_hz: float = Field(default=0.0, description="Synthetic frequency offset.")
    seed: Optional[int] = Field(default=None, description="RNG seed for reproducibility.")


# ─────────────────────────────────────────────────────────────────────────────
# Neural Model Config
# ─────────────────────────────────────────────────────────────────────────────

class ModelConfig(BaseModel):
    """Architecture hyperparameters for the Unified Neural Receiver."""

    # Input / framing
    chunk_size: int = Field(
        default=1024,
        description="IQ samples per inference chunk (power of 2).",
    )
    hop_size: int = Field(
        default=512,
        description="Hop between consecutive chunks (overlap = chunk-hop).",
    )

    # Encoder backbone
    resnet_channels: List[int] = Field(
        default=[64, 128, 256, 512],
        description="Channel widths for each ResNet1D stage.",
    )
    resnet_kernel: int = Field(default=7, description="Kernel size for residual blocks.")
    transformer_layers: int = Field(default=6, ge=1, le=24)
    transformer_heads: int = Field(default=8, ge=1)
    transformer_dim: int = Field(default=512, ge=64)
    transformer_ff_mult: int = Field(default=4, description="FFN expansion factor.")
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)

    # Conditioning / embedding
    mod_embed_dim: int = Field(
        default=128,
        description="Modulation embedding dimension (FiLM conditioning).",
    )
    freq_embed_dim: int = Field(
        default=64,
        description="Log-frequency embedding dimension.",
    )

    # AGC block
    agc_window: int = Field(default=64, description="AGC estimation window (samples).")
    agc_attack: float = Field(default=0.01, description="AGC attack rate (learned init).")
    agc_release: float = Field(default=0.001, description="AGC release rate (learned init).")

    # Channel filter
    filter_taps: int = Field(default=128, description="Neural filter kernel length.")
    filter_groups: int = Field(default=8, description="Depthwise groups in filter block.")

    # Freq offset corrector
    max_freq_offset_norm: float = Field(
        default=0.1,
        description="Max detectable freq offset as fraction of sample rate.",
    )

    # Heads
    analog_upsample_factor: int = Field(
        default=4,
        description="Upsampling ratio from encoder output to audio samples.",
    )
    digital_symbol_depth: int = Field(
        default=3,
        description="Number of MLP layers in the digital symbol head.",
    )

    # Weights
    weights_path: Optional[str] = Field(
        default=None,
        description="Path to pre-trained .pt checkpoint. None = random init.",
    )
    strict_load: bool = Field(
        default=False,
        description="Strict state_dict loading (False allows partial loads).",
    )

    @field_validator("chunk_size")
    @classmethod
    def must_be_power_of_two(cls, v: int) -> int:
        if v & (v - 1) != 0:
            raise ValueError(f"chunk_size must be a power of 2, got {v}")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Audio / Output Config
# ─────────────────────────────────────────────────────────────────────────────

class AudioConfig(BaseModel):
    output_sample_rate: int = Field(
        default=44100,
        description="PCM audio output sample rate (Hz).",
    )
    buffer_size: int = Field(default=2048, description="sounddevice buffer size.")
    volume: float = Field(default=0.8, ge=0.0, le=1.0)
    squelch_threshold: float = Field(
        default=0.3,
        ge=0.0, le=1.0,
        description="Squelch gate threshold (0=open, 1=closed). Neural head output.",
    )
    recording_path: Optional[str] = Field(
        default=None,
        description="If set, record demodulated audio to this WAV path.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# GUI Config
# ─────────────────────────────────────────────────────────────────────────────

class GUIConfig(BaseModel):
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=7860, ge=1024, le=65535)
    share: bool = Field(default=False, description="Create public Gradio share link.")
    waterfall_history: int = Field(
        default=200,
        description="Number of spectrum rows to keep in waterfall display.",
    )
    fft_size: int = Field(default=2048, description="FFT size for spectrum display.")
    update_interval_ms: int = Field(
        default=100,
        description="GUI refresh interval in milliseconds.",
    )
    theme: str = Field(default="dark", description="'dark' | 'light'.")


# ─────────────────────────────────────────────────────────────────────────────
# Training Config
# ─────────────────────────────────────────────────────────────────────────────

class TrainingConfig(BaseModel):
    batch_size: int = Field(default=64, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    epochs: int = Field(default=50, ge=1)
    warmup_steps: int = Field(default=1000, ge=0)
    grad_clip: float = Field(default=1.0, gt=0.0)
    val_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    checkpoint_dir: str = Field(default="weights/")
    log_every_n_steps: int = Field(default=50, ge=1)
    save_every_n_epochs: int = Field(default=5, ge=1)
    mixed_precision: bool = Field(default=True)

    # Synthetic data generation
    samples_per_epoch: int = Field(
        default=50_000,
        description="Number of IQ chunks generated per training epoch.",
    )
    snr_range_db: Tuple[float, float] = Field(
        default=(-10.0, 30.0),
        description="SNR range (dB) for synthetic data augmentation.",
    )
    freq_offset_range: Tuple[float, float] = Field(
        default=(-0.05, 0.05),
        description="Normalised freq offset range (fraction of sample rate).",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Top-Level Config
# ─────────────────────────────────────────────────────────────────────────────

class NeuralSDRConfig(BaseModel):
    """
    Master configuration object.

    Usage
    -----
    cfg = NeuralSDRConfig()                    # all defaults
    cfg = NeuralSDRConfig(center_freq=104.5e6) # override a field
    cfg.source_type = SourceType.SYNTHETIC     # mutate at runtime
    """
    model_config = {"arbitrary_types_allowed": True}

    # ── Signal parameters ─────────────────────────────────────────────────────
    center_freq: float = Field(
        default=100.0e6,
        ge=100e3,
        le=6.0e9,
        description="Center receive frequency (Hz).",
    )
    sample_rate: float = Field(
        default=2.048e6,
        gt=0.0,
        description="IQ sample rate (samples/sec).",
    )
    bandwidth: float = Field(
        default=200e3,
        gt=0.0,
        description="Desired signal bandwidth (Hz). NN filter is conditioned on this.",
    )
    modulation: ModulationMode = Field(
        default=ModulationMode.FM_WB,
        description="Active modulation mode — drives conditioning embedding.",
    )

    # ── Source ────────────────────────────────────────────────────────────────
    source_type: SourceType = Field(default=SourceType.SYNTHETIC)
    rtlsdr:    RTLSDRConfig    = Field(default_factory=RTLSDRConfig)
    soapy:     SoapyConfig     = Field(default_factory=SoapyConfig)
    websdr:    WebSDRConfig    = Field(default_factory=WebSDRConfig)
    file_src:  FileSourceConfig = Field(default_factory=FileSourceConfig)
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)

    # ── Sub-configs ───────────────────────────────────────────────────────────
    model:    ModelConfig    = Field(default_factory=ModelConfig)
    audio:    AudioConfig    = Field(default_factory=AudioConfig)
    gui:      GUIConfig      = Field(default_factory=GUIConfig)
    device:   DeviceConfig   = Field(default_factory=DeviceConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    # ── Runtime flags ─────────────────────────────────────────────────────────
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO", description="Loguru log level.")
    fine_tune_on_air: bool = Field(
        default=False,
        description="Enable online fine-tuning from live IQ stream (experimental).",
    )

    @model_validator(mode="after")
    def validate_bandwidth_vs_samplerate(self) -> "NeuralSDRConfig":
        if self.bandwidth > self.sample_rate:
            raise ValueError(
                f"bandwidth ({self.bandwidth/1e3:.1f} kHz) cannot exceed "
                f"sample_rate ({self.sample_rate/1e6:.3f} MHz)."
            )
        return self

    @model_validator(mode="after")
    def warn_analog_digital_mismatch(self) -> "NeuralSDRConfig":
        # Just a warning, not an error — user may know what they're doing
        return self

    def effective_chunk_duration_ms(self) -> float:
        return (self.model.chunk_size / self.sample_rate) * 1000.0

    def log_summary(self) -> None:
        logger.info("=" * 60)
        logger.info("NeuralSDR Configuration Summary")
        logger.info("=" * 60)
        logger.info(f"  Center Freq  : {self.center_freq/1e6:.6f} MHz")
        logger.info(f"  Sample Rate  : {self.sample_rate/1e6:.3f} MHz")
        logger.info(f"  Bandwidth    : {self.bandwidth/1e3:.1f} kHz")
        logger.info(f"  Modulation   : {self.modulation.value}")
        logger.info(f"  Source       : {self.source_type.value}")
        logger.info(f"  Chunk size   : {self.model.chunk_size} samples "
                    f"({self.effective_chunk_duration_ms():.2f} ms)")
        self.device.log_info()
        logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def default_config() -> NeuralSDRConfig:
    """Return a fully-default NeuralSDRConfig instance."""
    return NeuralSDRConfig()


def from_dict(d: dict) -> NeuralSDRConfig:
    """Construct config from a plain dictionary (e.g. loaded from JSON/YAML)."""
    return NeuralSDRConfig.model_validate(d)
