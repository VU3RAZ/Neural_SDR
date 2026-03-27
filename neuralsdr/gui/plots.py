"""
gui/plots.py
============
Plotly-based real-time plot renderers for the NeuralSDR GUI.

All functions return Plotly figure objects (or Pillow images for the
waterfall) that Gradio can display natively.

Plot types
----------
  make_spectrum_figure     — Single-shot power spectral density line plot
  make_waterfall_image     — Waterfall as a PIL Image (fast raster render)
  make_constellation_figure — IQ constellation scatter plot
  make_signal_meter_figure  — Analog-style signal strength gauge
  make_snr_bar             — Horizontal SNR bar chart
  make_bit_display         — Decoded bits as coloured boxes (digital modes)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

# Plotly is imported lazily to avoid import-time cost when not in GUI mode
_plotly_available = False
try:
    import plotly.graph_objects as go
    import plotly.express as px
    _plotly_available = True
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (dark SDR aesthetic)
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "bg":          "#0d1117",
    "surface":     "#161b22",
    "border":      "#30363d",
    "text":        "#e6edf3",
    "text_dim":    "#8b949e",
    "accent":      "#58a6ff",
    "accent2":     "#3fb950",
    "warn":        "#d29922",
    "danger":      "#f85149",
    "spectrum":    "#00d4ff",
    "waterfall_lo": [0,   0,   40],   # dark blue  (low power)
    "waterfall_hi": [255, 220, 0],    # yellow     (high power)
}

_LAYOUT_DEFAULTS = dict(
    paper_bgcolor=PALETTE["bg"],
    plot_bgcolor =PALETTE["surface"],
    font=dict(color=PALETTE["text"], size=11, family="monospace"),
    margin=dict(l=50, r=20, t=30, b=40),
)


def _axis_style(**kwargs):
    return dict(
        gridcolor=PALETTE["border"],
        zerolinecolor=PALETTE["border"],
        tickfont=dict(color=PALETTE["text_dim"], size=10),
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Spectrum plot
# ─────────────────────────────────────────────────────────────────────────────

def make_spectrum_figure(
    freqs_mhz:    np.ndarray,
    psd_db:       np.ndarray,
    center_freq_mhz: float = 100.0,
    title:        str = "Spectrum",
    height:       int = 220,
) -> "go.Figure":
    """
    Power spectral density line plot (DC-centred).

    Parameters
    ----------
    freqs_mhz    : np.ndarray  float32  [N]  — frequency axis in MHz
    psd_db       : np.ndarray  float32  [N]  — power in dBFS
    center_freq_mhz : float — for axis formatting
    title        : str
    height       : int

    Returns
    -------
    plotly Figure
    """
    if not _plotly_available:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs_mhz,
        y=psd_db,
        mode="lines",
        line=dict(color=PALETTE["spectrum"], width=1.2),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.08)",
        name="PSD",
        hovertemplate="%{x:.4f} MHz<br>%{y:.1f} dBFS<extra></extra>",
    ))

    # Vertical line at centre frequency
    fig.add_vline(
        x=center_freq_mhz,
        line=dict(color=PALETTE["accent"], width=1, dash="dot"),
    )

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(text=title, font=dict(size=12, color=PALETTE["text_dim"])),
        height=height,
        showlegend=False,
        xaxis=_axis_style(title="Frequency (MHz)", tickformat=".3f"),
        yaxis=_axis_style(title="dBFS", range=[-100, 0]),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Waterfall image (PIL for performance)
# ─────────────────────────────────────────────────────────────────────────────

_WATERFALL_COLORMAP: Optional[np.ndarray] = None


def _get_waterfall_colormap() -> np.ndarray:
    """Build a 256-entry RGB LUT for the waterfall (black → blue → cyan → yellow)."""
    global _WATERFALL_COLORMAP
    if _WATERFALL_COLORMAP is not None:
        return _WATERFALL_COLORMAP

    lut = np.zeros((256, 3), dtype=np.uint8)
    # Segment 0–85:   black → dark blue
    for i in range(86):
        t = i / 85.0
        lut[i] = [0, 0, int(100 * t)]
    # Segment 85–170: dark blue → cyan
    for i in range(86, 171):
        t = (i - 86) / 85.0
        lut[i] = [0, int(200 * t), int(100 + 155 * t)]
    # Segment 170–255: cyan → yellow
    for i in range(171, 256):
        t = (i - 171) / 85.0
        lut[i] = [int(255 * t), int(200 + 55 * t), int(255 * (1 - t))]

    _WATERFALL_COLORMAP = lut
    return lut


def make_waterfall_image(
    waterfall_norm: np.ndarray,
    width:  int = 800,
    height: int = 300,
) -> Image.Image:
    """
    Convert a normalised waterfall array to a PIL RGB Image.

    Parameters
    ----------
    waterfall_norm : np.ndarray  float32  shape (H, W)  values in [0, 1]
                     H = history rows, W = FFT bins (newest row at top)
    width          : int — Output image width in pixels
    height         : int — Output image height in pixels

    Returns
    -------
    PIL.Image.Image  RGB
    """
    lut = _get_waterfall_colormap()

    # Scale to 0–255 and apply colormap
    idx = (waterfall_norm * 255).clip(0, 255).astype(np.uint8)
    rgb = lut[idx]   # (H, W, 3)

    img = Image.fromarray(rgb, mode="RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.NEAREST)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Constellation plot
# ─────────────────────────────────────────────────────────────────────────────

# Expected constellation geometry for visual reference overlays
_CONST_REFS = {
    "BPSK":  [(-1, 0), (1, 0)],
    "QPSK":  [( 0.707,  0.707), (-0.707,  0.707),
               (-0.707, -0.707), ( 0.707, -0.707)],
    "8PSK":  [(math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)) for k in range(8)],
}


def make_constellation_figure(
    i_points:  np.ndarray,
    q_points:  np.ndarray,
    mod_name:  str = "",
    max_pts:   int = 512,
    height:    int = 300,
) -> "go.Figure":
    """
    IQ constellation scatter plot.

    Parameters
    ----------
    i_points : np.ndarray  float32  [N]  — I (real) component
    q_points : np.ndarray  float32  [N]  — Q (imaginary) component
    mod_name : str — Modulation name for overlay reference points
    max_pts  : int — Maximum scatter points to show (downsamples if needed)
    height   : int

    Returns
    -------
    plotly Figure
    """
    if not _plotly_available:
        return None

    # Downsample
    if len(i_points) > max_pts:
        idx = np.random.choice(len(i_points), max_pts, replace=False)
        i_points = i_points[idx]
        q_points = q_points[idx]

    # Colour by density (intensity = index recency, newer = brighter)
    n = len(i_points)
    colors = np.linspace(0.3, 1.0, n)

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=i_points.tolist(),
        y=q_points.tolist(),
        mode="markers",
        marker=dict(
            size=3,
            color=colors,
            colorscale=[[0, "rgba(0,100,200,0.2)"], [1, "rgba(0,220,255,0.9)"]],
            showscale=False,
        ),
        name="IQ",
        hoverinfo="skip",
    ))

    # Reference constellation overlay
    refs = _CONST_REFS.get(mod_name.upper(), [])
    if refs:
        rx, ry = zip(*refs)
        fig.add_trace(go.Scatter(
            x=list(rx), y=list(ry),
            mode="markers",
            marker=dict(
                size=8, color=PALETTE["accent"], symbol="x",
                line=dict(width=1, color=PALETTE["accent"]),
            ),
            name="Ideal",
        ))

    # Unit circle reference
    theta = np.linspace(0, 2 * np.pi, 128)
    fig.add_trace(go.Scatter(
        x=np.cos(theta).tolist(), y=np.sin(theta).tolist(),
        mode="lines",
        line=dict(color=PALETTE["border"], width=0.8, dash="dot"),
        name="Unit circle",
        hoverinfo="skip",
    ))

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        height=height,
        title=dict(
            text=f"Constellation  {mod_name}  (n={n})",
            font=dict(size=11, color=PALETTE["text_dim"]),
        ),
        showlegend=False,
        xaxis=_axis_style(title="I", range=[-1.8, 1.8], scaleanchor="y"),
        yaxis=_axis_style(title="Q", range=[-1.8, 1.8]),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Signal strength gauge
# ─────────────────────────────────────────────────────────────────────────────

def make_signal_meter_figure(
    level_db:       float,
    squelch_db:     float = -40.0,
    min_db:         float = -100.0,
    max_db:         float = 0.0,
    height:         int   = 90,
    squelch_open:   bool  = True,
) -> "go.Figure":
    """
    Horizontal signal strength bar with squelch threshold marker.

    Parameters
    ----------
    level_db     : float — Current signal level in dBFS.
    squelch_db   : float — Squelch threshold in dBFS for display.
    min_db, max_db: float — Display range.
    height       : int
    squelch_open : bool — Whether the squelch is currently open.

    Returns
    -------
    plotly Figure
    """
    if not _plotly_available:
        return None

    level_norm = (level_db - min_db) / (max_db - min_db)
    level_norm = float(np.clip(level_norm, 0.0, 1.0))
    squelch_norm = (squelch_db - min_db) / (max_db - min_db)
    squelch_norm = float(np.clip(squelch_norm, 0.0, 1.0))

    bar_color = PALETTE["accent2"] if squelch_open else PALETTE["warn"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[level_norm], y=["Signal"],
        orientation="h",
        marker=dict(
            color=bar_color,
            line=dict(color=bar_color, width=0),
        ),
        width=0.5,
        hovertemplate=f"{level_db:.1f} dBFS<extra></extra>",
    ))

    # Squelch threshold line
    fig.add_vline(
        x=squelch_norm,
        line=dict(color=PALETTE["warn"], width=2, dash="dash"),
        annotation_text="SQL",
        annotation_font=dict(size=9, color=PALETTE["warn"]),
    )

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        height=height,
        showlegend=False,
        xaxis=dict(
            range=[0, 1],
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ticktext=[f"{min_db:.0f}", f"{(min_db+max_db)*0.25:.0f}",
                      f"{(min_db+max_db)*0.5:.0f}", f"{max_db*0.75:.0f}", f"{max_db:.0f}"],
            gridcolor=PALETTE["border"],
            tickfont=dict(color=PALETTE["text_dim"], size=9),
        ),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=5, b=25),
        annotations=[dict(
            x=level_norm, y=0.9,
            text=f"{level_db:.1f} dBFS",
            showarrow=False,
            font=dict(color=PALETTE["text"], size=11),
            xanchor="left" if level_norm < 0.8 else "right",
        )],
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SNR estimate bar
# ─────────────────────────────────────────────────────────────────────────────

def make_snr_bar(
    snr_db: float,
    height: int = 80,
) -> "go.Figure":
    """Horizontal SNR indicator bar, coloured by quality."""
    if not _plotly_available:
        return None

    snr_clamp = float(np.clip(snr_db, -20, 50))
    norm = (snr_clamp + 20) / 70.0

    if snr_clamp < 5:
        color = PALETTE["danger"]
    elif snr_clamp < 15:
        color = PALETTE["warn"]
    else:
        color = PALETTE["accent2"]

    fig = go.Figure(go.Bar(
        x=[norm], y=["SNR"],
        orientation="h",
        marker=dict(color=color),
        width=0.5,
    ))
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        height=height,
        showlegend=False,
        xaxis=dict(range=[0, 1], tickvals=[0, 0.5, 1],
                   ticktext=["-20", "15", "50"],
                   gridcolor=PALETTE["border"],
                   tickfont=dict(color=PALETTE["text_dim"], size=9)),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=5, b=25),
        annotations=[dict(
            x=max(0.05, norm), y=0.9,
            text=f"{snr_clamp:.1f} dB",
            showarrow=False,
            font=dict(color=PALETTE["text"], size=11),
        )],
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Decoded bits display (digital modes)
# ─────────────────────────────────────────────────────────────────────────────

def make_bit_display_figure(
    bits: np.ndarray,
    max_bits: int = 128,
    height: int = 80,
) -> "go.Figure":
    """
    Render decoded bits as a row of coloured cells (0=dark, 1=bright).

    Parameters
    ----------
    bits     : np.ndarray  float32  [N]  — decoded bits (0 or 1)
    max_bits : int
    height   : int
    """
    if not _plotly_available:
        return None

    bits = np.asarray(bits[:max_bits], dtype=np.float32)
    n = len(bits)
    if n == 0:
        return None

    colors = [PALETTE["accent2"] if b > 0.5 else PALETTE["surface"] for b in bits]
    texts  = ["1" if b > 0.5 else "0" for b in bits]

    fig = go.Figure(go.Bar(
        x=list(range(n)),
        y=[1] * n,
        marker=dict(
            color=colors,
            line=dict(color=PALETTE["border"], width=0.5),
        ),
        text=texts,
        textposition="inside",
        textfont=dict(size=7, color=PALETTE["text"]),
        hoverinfo="skip",
    ))

    ber_approx = 1 - float(bits.mean()) if bits.mean() > 0.5 else float(bits.mean())
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        height=height,
        showlegend=False,
        title=dict(
            text=f"Decoded bits (n={n}, approx BER≈{ber_approx:.3f})",
            font=dict(size=10, color=PALETTE["text_dim"]),
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, range=[0, 1.5]),
        margin=dict(l=5, r=5, t=22, b=5),
        bargap=0.05,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics / info panel
# ─────────────────────────────────────────────────────────────────────────────

def format_diagnostics(
    freq_offset_norm: float,
    phase_offset_rad: float,
    sample_rate:      float,
    snr_db:           float,
    presence_prob:    float,
    mod_name:         str,
    source_name:      str,
) -> str:
    """
    Format a Markdown diagnostics string for the info panel.

    Returns
    -------
    str  — Markdown-formatted diagnostics text.
    """
    freq_offset_hz = freq_offset_norm * sample_rate
    squelch_emoji = "🟢" if presence_prob > 0.5 else "🔴"

    return (
        f"| Parameter | Value |\n"
        f"|-----------|-------|\n"
        f"| Mode | **{mod_name}** |\n"
        f"| Source | {source_name} |\n"
        f"| Freq offset | {freq_offset_hz:+.1f} Hz ({freq_offset_norm*1e6:+.1f} ppm) |\n"
        f"| Phase offset | {math.degrees(phase_offset_rad):+.1f}° |\n"
        f"| Est. SNR | {snr_db:.1f} dB |\n"
        f"| Squelch | {squelch_emoji} {'Open' if presence_prob > 0.5 else 'Closed'} "
        f"(p={presence_prob:.2f}) |\n"
    )
