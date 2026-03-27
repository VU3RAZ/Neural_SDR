"""
sources/__init__.py
===================
IQ source factory and public API.
"""

from .base_source import IQSource
from .rtlsdr_source import RTLSDRSource
from .soapy_source import SoapySDRSource
from .websdr_source import WebSDRSource
from .file_source import FileSource
from .synthetic_source import SyntheticSource, generate_iq

from ..config.settings import NeuralSDRConfig, SourceType


def build_source(config: NeuralSDRConfig) -> IQSource:
    """
    Factory: construct the correct IQSource subclass from a NeuralSDRConfig.

    Parameters
    ----------
    config : NeuralSDRConfig

    Returns
    -------
    IQSource — not yet opened; call .open() or use as context manager.
    """
    stype = config.source_type
    fs = config.sample_rate
    fc = config.center_freq

    if stype == SourceType.RTL_SDR:
        return RTLSDRSource(fs, fc, config.rtlsdr)

    elif stype == SourceType.SOAPY:
        return SoapySDRSource(fs, fc, config.soapy)

    elif stype == SourceType.WEBSDR:
        return WebSDRSource(fs, fc, config.websdr)

    elif stype == SourceType.FILE:
        return FileSource(fs, fc, config.file_src)

    elif stype == SourceType.SYNTHETIC:
        device = config.device.device
        return SyntheticSource(fs, fc, config.modulation, config.synthetic, device)

    else:
        raise ValueError(f"Unknown source type: {stype!r}")


__all__ = [
    "IQSource",
    "RTLSDRSource",
    "SoapySDRSource",
    "WebSDRSource",
    "FileSource",
    "SyntheticSource",
    "generate_iq",
    "build_source",
]
