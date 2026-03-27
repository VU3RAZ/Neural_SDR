"""
config/presets.py
=================
Predefined frequency / modulation presets for common broadcast and amateur bands.

Each preset fully specifies the signal parameters that will be applied when the
user clicks a quick-access button in the GUI.  Presets are intentionally kept as
plain dicts so they can easily be serialised / extended by end users.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BandPreset:
    """
    A single ready-to-use receive configuration.

    Parameters
    ----------
    name        : Display name shown on the GUI button.
    category    : Grouping label ('Broadcast', 'Amateur', 'Utility', …).
    center_freq : Centre frequency in Hz.
    sample_rate : IQ sample rate in samples/sec.
    bandwidth   : Signal bandwidth in Hz.
    modulation  : ModulationMode value string (must match ModulationMode enum).
    description : One-line human-readable note.
    tags        : Arbitrary searchable tags.
    """
    name:        str
    category:    str
    center_freq: float
    sample_rate: float
    bandwidth:   float
    modulation:  str
    description: str        = ""
    tags:        List[str]  = field(default_factory=list)


@dataclass(frozen=True)
class ModPreset:
    """
    A modulation-only preset (no frequency).
    Used to quickly switch demodulation mode while keeping the current frequency.
    """
    name:       str
    modulation: str
    bandwidth:  float
    description: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Band presets
# ─────────────────────────────────────────────────────────────────────────────

BAND_PRESETS: Dict[str, BandPreset] = {

    # ── FM Broadcast ──────────────────────────────────────────────────────────
    "fm_broadcast_typical": BandPreset(
        name        = "FM 100 MHz",
        category    = "Broadcast",
        center_freq = 100.0e6,
        sample_rate = 2.048e6,
        bandwidth   = 200e3,
        modulation  = "FM_WB",
        description = "Typical FM broadcast station",
        tags        = ["fm", "broadcast", "audio"],
    ),
    "fm_broadcast_87_5": BandPreset(
        name        = "FM 87.5 MHz",
        category    = "Broadcast",
        center_freq = 87.5e6,
        sample_rate = 2.048e6,
        bandwidth   = 200e3,
        modulation  = "FM_WB",
        description = "Bottom of FM broadcast band",
        tags        = ["fm", "broadcast"],
    ),
    "fm_broadcast_107_9": BandPreset(
        name        = "FM 107.9 MHz",
        category    = "Broadcast",
        center_freq = 107.9e6,
        sample_rate = 2.048e6,
        bandwidth   = 200e3,
        modulation  = "FM_WB",
        description = "Top of FM broadcast band",
        tags        = ["fm", "broadcast"],
    ),

    # ── AM Broadcast ──────────────────────────────────────────────────────────
    "am_broadcast_medium_wave": BandPreset(
        name        = "AM 1000 kHz",
        category    = "Broadcast",
        center_freq = 1.0e6,
        sample_rate = 2.048e6,
        bandwidth   = 10e3,
        modulation  = "AM",
        description = "Medium-wave AM broadcast",
        tags        = ["am", "broadcast", "mw"],
    ),
    "am_broadcast_sw_25m": BandPreset(
        name        = "SW 25m (11.6 MHz)",
        category    = "Broadcast",
        center_freq = 11.6e6,
        sample_rate = 2.048e6,
        bandwidth   = 10e3,
        modulation  = "AM",
        description = "25-metre shortwave broadcast band",
        tags        = ["am", "shortwave", "sw"],
    ),
    "am_broadcast_sw_49m": BandPreset(
        name        = "SW 49m (6.0 MHz)",
        category    = "Broadcast",
        center_freq = 6.0e6,
        sample_rate = 2.048e6,
        bandwidth   = 10e3,
        modulation  = "AM",
        description = "49-metre shortwave broadcast band",
        tags        = ["am", "shortwave", "sw"],
    ),

    # ── Aviation ──────────────────────────────────────────────────────────────
    "aviation_unicom": BandPreset(
        name        = "Aviation UNICOM 123.0 MHz",
        category    = "Aviation",
        center_freq = 123.0e6,
        sample_rate = 2.048e6,
        bandwidth   = 25e3,
        modulation  = "AM",
        description = "Aviation UNICOM / general aviation",
        tags        = ["aviation", "am", "vhf"],
    ),
    "aviation_ground": BandPreset(
        name        = "Aviation 121.5 MHz (Guard)",
        category    = "Aviation",
        center_freq = 121.5e6,
        sample_rate = 2.048e6,
        bandwidth   = 25e3,
        modulation  = "AM",
        description = "International distress / guard frequency",
        tags        = ["aviation", "am", "guard"],
    ),
    "vor_navigation": BandPreset(
        name        = "VOR 112.0 MHz",
        category    = "Aviation",
        center_freq = 112.0e6,
        sample_rate = 2.048e6,
        bandwidth   = 50e3,
        modulation  = "AM",
        description = "VHF Omnidirectional Range navigation",
        tags        = ["aviation", "vor", "navigation"],
    ),

    # ── Amateur Radio ─────────────────────────────────────────────────────────
    "ham_40m_ssb": BandPreset(
        name        = "40m Ham (7.2 MHz) SSB",
        category    = "Amateur",
        center_freq = 7.2e6,
        sample_rate = 2.048e6,
        bandwidth   = 3e3,
        modulation  = "USB",
        description = "40-metre ham band USB phone segment",
        tags        = ["ham", "ssb", "hf", "40m"],
    ),
    "ham_20m_ssb": BandPreset(
        name        = "20m Ham (14.225 MHz) SSB",
        category    = "Amateur",
        center_freq = 14.225e6,
        sample_rate = 2.048e6,
        bandwidth   = 3e3,
        modulation  = "USB",
        description = "20-metre ham band USB (DX phone)",
        tags        = ["ham", "ssb", "hf", "20m", "dx"],
    ),
    "ham_20m_cw": BandPreset(
        name        = "20m Ham CW (14.025 MHz)",
        category    = "Amateur",
        center_freq = 14.025e6,
        sample_rate = 2.048e6,
        bandwidth   = 500,
        modulation  = "CW",
        description = "20-metre CW segment",
        tags        = ["ham", "cw", "morse", "hf", "20m"],
    ),
    "ham_2m_fm": BandPreset(
        name        = "2m Ham (145.5 MHz) FM",
        category    = "Amateur",
        center_freq = 145.5e6,
        sample_rate = 2.048e6,
        bandwidth   = 16e3,
        modulation  = "FM_NB",
        description = "2-metre amateur FM simplex calling",
        tags        = ["ham", "fm", "vhf", "2m"],
    ),
    "ham_70cm_fm": BandPreset(
        name        = "70cm Ham (433.5 MHz) FM",
        category    = "Amateur",
        center_freq = 433.5e6,
        sample_rate = 2.048e6,
        bandwidth   = 16e3,
        modulation  = "FM_NB",
        description = "70-centimetre amateur FM",
        tags        = ["ham", "fm", "uhf", "70cm"],
    ),
    "ham_bpsk31_20m": BandPreset(
        name        = "BPSK-31 (14.070 MHz)",
        category    = "Amateur",
        center_freq = 14.070e6,
        sample_rate = 2.048e6,
        bandwidth   = 3e3,
        modulation  = "BPSK",
        description = "20m BPSK-31 digital segment",
        tags        = ["ham", "digital", "bpsk", "hf", "20m"],
    ),

    # ── Marine ────────────────────────────────────────────────────────────────
    "marine_ch16_distress": BandPreset(
        name        = "Marine VHF Ch16 (156.8 MHz)",
        category    = "Marine",
        center_freq = 156.8e6,
        sample_rate = 2.048e6,
        bandwidth   = 25e3,
        modulation  = "FM_NB",
        description = "International maritime distress / calling",
        tags        = ["marine", "fm", "vhf", "distress"],
    ),
    "marine_ais_161": BandPreset(
        name        = "AIS 161.975 MHz",
        category    = "Marine",
        center_freq = 161.975e6,
        sample_rate = 2.048e6,
        bandwidth   = 25e3,
        modulation  = "GFSK",
        description = "Automatic Identification System (AIS) channel 87B",
        tags        = ["marine", "ais", "digital", "gfsk"],
    ),

    # ── Public Safety / PMR ───────────────────────────────────────────────────
    "pmr446": BandPreset(
        name        = "PMR446 (446.1 MHz)",
        category    = "Utility",
        center_freq = 446.1e6,
        sample_rate = 2.048e6,
        bandwidth   = 16e3,
        modulation  = "FM_NB",
        description = "European licence-free PMR446 walkie-talkie band",
        tags        = ["pmr", "fm", "uhf"],
    ),
    "gmrs_462": BandPreset(
        name        = "GMRS 462.5 MHz",
        category    = "Utility",
        center_freq = 462.5e6,
        sample_rate = 2.048e6,
        bandwidth   = 25e3,
        modulation  = "FM_NB",
        description = "US General Mobile Radio Service",
        tags        = ["gmrs", "fm", "uhf"],
    ),

    # ── ISM / IoT ─────────────────────────────────────────────────────────────
    "ism_433": BandPreset(
        name        = "ISM 433.92 MHz",
        category    = "ISM / IoT",
        center_freq = 433.92e6,
        sample_rate = 2.048e6,
        bandwidth   = 200e3,
        modulation  = "GFSK",
        description = "European ISM band — sensors, remotes, LoRa",
        tags        = ["ism", "iot", "lora", "gfsk"],
    ),
    "ism_868": BandPreset(
        name        = "ISM 868 MHz",
        category    = "ISM / IoT",
        center_freq = 868.0e6,
        sample_rate = 2.048e6,
        bandwidth   = 500e3,
        modulation  = "GFSK",
        description = "European LoRaWAN / Sigfox uplink band",
        tags        = ["ism", "iot", "lorawan", "gfsk"],
    ),
    "ism_915": BandPreset(
        name        = "ISM 915 MHz",
        category    = "ISM / IoT",
        center_freq = 915.0e6,
        sample_rate = 2.048e6,
        bandwidth   = 500e3,
        modulation  = "GFSK",
        description = "US ISM band — LoRaWAN, 802.15.4g",
        tags        = ["ism", "iot", "lorawan", "gfsk"],
    ),
    "ism_2400_bluetooth": BandPreset(
        name        = "Bluetooth 2.402 GHz",
        category    = "ISM / IoT",
        center_freq = 2.402e9,
        sample_rate = 4.0e6,
        bandwidth   = 1e6,
        modulation  = "GFSK",
        description = "Bluetooth Classic / BLE channel 37",
        tags        = ["bluetooth", "ble", "gfsk", "ism"],
    ),

    # ── Weather ───────────────────────────────────────────────────────────────
    "noaa_wx_162_400": BandPreset(
        name        = "NOAA WX 162.400 MHz",
        category    = "Weather",
        center_freq = 162.400e6,
        sample_rate = 2.048e6,
        bandwidth   = 25e3,
        modulation  = "FM_NB",
        description = "NOAA Weather Radio channel WX1",
        tags        = ["noaa", "weather", "fm", "vhf"],
    ),

    # ── Space / Satellite ─────────────────────────────────────────────────────
    "noaa_apt_137_500": BandPreset(
        name        = "NOAA APT 137.5 MHz",
        category    = "Satellite",
        center_freq = 137.5e6,
        sample_rate = 2.048e6,
        bandwidth   = 40e3,
        modulation  = "AM",
        description = "NOAA POES satellite APT image downlink",
        tags        = ["noaa", "satellite", "apt", "image"],
    ),
    "iss_voice": BandPreset(
        name        = "ISS Voice 145.800 MHz",
        category    = "Satellite",
        center_freq = 145.800e6,
        sample_rate = 2.048e6,
        bandwidth   = 16e3,
        modulation  = "FM_NB",
        description = "International Space Station FM voice downlink",
        tags        = ["iss", "satellite", "fm", "amateur"],
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Modulation presets (mode-only, no frequency)
# ─────────────────────────────────────────────────────────────────────────────

MOD_PRESETS: Dict[str, ModPreset] = {
    "am_voice": ModPreset(
        name       = "AM Voice",
        modulation = "AM",
        bandwidth  = 10e3,
        description = "Standard AM broadcast / aviation voice",
    ),
    "fm_narrowband": ModPreset(
        name       = "FM Narrowband",
        modulation = "FM_NB",
        bandwidth  = 16e3,
        description = "NFM for amateur / utility VHF/UHF",
    ),
    "fm_wideband": ModPreset(
        name       = "FM Wideband",
        modulation = "FM_WB",
        bandwidth  = 200e3,
        description = "WFM for FM broadcast",
    ),
    "usb_voice": ModPreset(
        name       = "USB Voice",
        modulation = "USB",
        bandwidth  = 3e3,
        description = "Upper Sideband SSB for HF phone",
    ),
    "lsb_voice": ModPreset(
        name       = "LSB Voice",
        modulation = "LSB",
        bandwidth  = 3e3,
        description = "Lower Sideband SSB — HF below 10 MHz",
    ),
    "cw_narrow": ModPreset(
        name       = "CW Narrow",
        modulation = "CW",
        bandwidth  = 500,
        description = "CW with 500 Hz bandwidth",
    ),
    "bpsk31": ModPreset(
        name       = "BPSK-31",
        modulation = "BPSK",
        bandwidth  = 100,
        description = "BPSK at 31.25 baud (slow, robust)",
    ),
    "qpsk": ModPreset(
        name       = "QPSK",
        modulation = "QPSK",
        bandwidth  = 25e3,
        description = "Generic QPSK (satellite, data links)",
    ),
    "qam16": ModPreset(
        name       = "16-QAM",
        modulation = "16QAM",
        bandwidth  = 200e3,
        description = "16-QAM — cable/broadband-like",
    ),
    "gfsk_ism": ModPreset(
        name       = "GFSK ISM",
        modulation = "GFSK",
        bandwidth  = 200e3,
        description = "GFSK for ISM band sensors and LoRa",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Lookup helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_preset(key: str) -> Optional[BandPreset]:
    """Return a BandPreset by key, or None if not found."""
    return BAND_PRESETS.get(key)


def list_presets_by_category() -> Dict[str, List[BandPreset]]:
    """Return all band presets grouped by category."""
    result: Dict[str, List[BandPreset]] = {}
    for preset in BAND_PRESETS.values():
        result.setdefault(preset.category, []).append(preset)
    return result


def search_presets(query: str) -> List[BandPreset]:
    """
    Simple fuzzy text search over name, description, and tags.
    Returns presets whose name, description, or any tag contains `query`
    (case-insensitive).
    """
    q = query.lower()
    results = []
    for preset in BAND_PRESETS.values():
        haystack = (
            preset.name.lower()
            + " " + preset.description.lower()
            + " " + " ".join(preset.tags)
        )
        if q in haystack:
            results.append(preset)
    return results


def apply_preset_to_config(
    preset: BandPreset,
    config,           # NeuralSDRConfig — avoid circular import with string hint
) -> None:
    """
    Mutate a NeuralSDRConfig in-place to match a BandPreset.

    Parameters
    ----------
    preset : BandPreset
    config : NeuralSDRConfig
    """
    from .settings import ModulationMode  # local import to avoid circular deps

    config.center_freq = preset.center_freq
    config.sample_rate = preset.sample_rate
    config.bandwidth   = preset.bandwidth
    config.modulation  = ModulationMode(preset.modulation)


# ─────────────────────────────────────────────────────────────────────────────
# Sample rate catalogue — values shown in GUI dropdown
# ─────────────────────────────────────────────────────────────────────────────

COMMON_SAMPLE_RATES = [
    (  "240 kS/s",   240_000),
    (  "960 kS/s",   960_000),
    ("1.024 MS/s", 1_024_000),
    ("1.200 MS/s", 1_200_000),
    ("1.440 MS/s", 1_440_000),
    ("2.048 MS/s", 2_048_000),
    ("2.400 MS/s", 2_400_000),
    ("3.200 MS/s", 3_200_000),
    ("4.000 MS/s", 4_000_000),
    ("8.000 MS/s", 8_000_000),
    ("10.00 MS/s", 10_000_000),
    ("20.00 MS/s", 20_000_000),
]

COMMON_BANDWIDTHS = [
    (     "500 Hz",     500),
    (   "2.5 kHz",   2_500),
    (   "3.0 kHz",   3_000),
    (   "6.0 kHz",   6_000),
    (  "10.0 kHz",  10_000),
    (  "16.0 kHz",  16_000),
    (  "25.0 kHz",  25_000),
    (  "50.0 kHz",  50_000),
    ( "100.0 kHz", 100_000),
    ( "200.0 kHz", 200_000),
    ( "500.0 kHz", 500_000),
    (   "1.0 MHz", 1_000_000),
]
