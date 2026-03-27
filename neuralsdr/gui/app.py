"""
gui/app.py
==========
NeuralSDR Gradio web interface.

Provides a complete real-time SDR receiver GUI with:
  - Waterfall display
  - Power spectrum plot
  - IQ constellation viewer
  - Signal strength / SNR meters
  - Decoded bits display (digital modes)
  - Diagnostics panel
  - Full receive parameter controls
  - Quick-access band presets
  - Audio playback controls
  - Recording toggle

Architecture
------------
A single background thread runs the SDR receiver pipeline and updates
a shared state dictionary.  Gradio's streaming interface polls this
state at ~10 Hz to refresh the visual components.

State dictionary (thread-shared)
---------------------------------
  _state = {
    "spectrum":      (freqs_mhz, psd_db),
    "waterfall":     waterfall_norm_array,
    "constellation": (i_pts, q_pts),
    "signal_db":     float,
    "snr_db":        float,
    "presence_prob": float,
    "freq_offset":   float,
    "phase_offset":  float,
    "bits":          np.ndarray or None,
    "audio":         np.ndarray or None,
    "running":       bool,
    "error":         str or None,
  }
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Optional

import numpy as np
from loguru import logger

try:
    import gradio as gr
    _gradio_available = True
except ImportError:
    _gradio_available = False
    logger.warning("Gradio not installed.  Run: pip install gradio")

from ..config.settings  import NeuralSDRConfig, ModulationMode, SourceType
from ..config.presets   import (
    BAND_PRESETS, list_presets_by_category,
    apply_preset_to_config, COMMON_SAMPLE_RATES,
)
from ..dsp.fft_utils    import (
    power_spectrum_fast, WaterfallAccumulator,
    freq_axis_mhz, signal_strength_dbfs,
    extract_constellation_points,
)
from ..dsp.iq_utils     import iq_to_tensor
from ..sources          import build_source
from ..audio.output     import make_audio_output
from .plots import (
    make_spectrum_figure, make_waterfall_image,
    make_constellation_figure, make_signal_meter_figure,
    make_snr_bar, make_bit_display_figure, format_diagnostics,
)


# ─────────────────────────────────────────────────────────────────────────────
# Receiver engine (background thread)
# ─────────────────────────────────────────────────────────────────────────────

class ReceiverEngine:
    """
    Background thread that runs the full SDR pipeline and updates shared state.

    Parameters
    ----------
    cfg          : NeuralSDRConfig
    neural_model : NeuralReceiver (already built + on device)
    audio_output : AudioOutput
    """

    def __init__(self, cfg: NeuralSDRConfig, neural_model, audio_output) -> None:
        self.cfg           = cfg
        self.model         = neural_model
        self.audio_output  = audio_output

        self._waterfall = WaterfallAccumulator(
            fft_size=cfg.gui.fft_size,
            history=cfg.gui.waterfall_history,
        )

        # Shared state dict (read by GUI polling loop)
        self.state = {
            "spectrum":      (np.array([0.0]), np.array([-100.0])),
            "waterfall":     np.zeros((cfg.gui.waterfall_history, cfg.gui.fft_size)),
            "constellation": (np.array([0.0]), np.array([0.0])),
            "signal_db":     -100.0,
            "snr_db":        0.0,
            "presence_prob": 0.0,
            "freq_offset":   0.0,
            "phase_offset":  0.0,
            "bits":          None,
            "audio_chunk":   None,
            "running":       False,
            "error":         None,
            "chunks_processed": 0,
        }
        self._lock   = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop   = threading.Event()

    # ── Start / stop ──────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="receiver-engine", daemon=True
        )
        self._thread.start()
        logger.info("[ReceiverEngine] Started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._update_state({"running": False})
        logger.info("[ReceiverEngine] Stopped")

    def update_config(self, new_cfg: NeuralSDRConfig) -> None:
        """Hot-update config (frequency, modulation, etc.)."""
        self.cfg = new_cfg

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _run(self) -> None:
        import torch
        self._update_state({"running": True, "error": None})
        chunk_size = self.cfg.model.chunk_size
        device     = self.cfg.device.device

        try:
            source = build_source(self.cfg)
            source.open()

            with source:
                while not self._stop.is_set():
                    # Read IQ chunk
                    iq_np = source.read_samples(chunk_size)

                    # ── DSP display path (spectrum, waterfall) ─────────────
                    self._waterfall.push(iq_np)
                    psd_db = power_spectrum_fast(iq_np, self.cfg.gui.fft_size)
                    freqs_mhz = freq_axis_mhz(
                        self.cfg.center_freq,
                        self.cfg.sample_rate,
                        self.cfg.gui.fft_size,
                    )
                    sig_db = signal_strength_dbfs(iq_np)
                    i_pts, q_pts = extract_constellation_points(iq_np, sps=8)

                    # ── Neural inference ───────────────────────────────────
                    iq_t = iq_to_tensor(iq_np, device=device).unsqueeze(0)
                    mod  = self.cfg.modulation
                    mod_idx = torch.tensor([mod.index], device=device)
                    cf      = torch.tensor([self.cfg.center_freq], device=device)
                    bw      = torch.tensor([self.cfg.bandwidth], device=device)

                    with torch.no_grad():
                        output = self.model(
                            iq_t, mod_idx, cf, bw,
                            mod_mode    = mod,
                            run_analog  = mod.is_analog,
                            run_digital = mod.is_digital,
                        )

                    # ── Extract outputs ────────────────────────────────────
                    presence_prob = float(output.presence_prob[0].item()) \
                        if output.presence_prob is not None else 1.0
                    snr_db = float(output.snr_db[0].item()) \
                        if output.snr_db is not None else 0.0
                    freq_off = float(output.freq_offset_norm[0].item()) \
                        if output.freq_offset_norm is not None else 0.0
                    phase_off = float(output.phase_offset_rad[0].item()) \
                        if output.phase_offset_rad is not None else 0.0

                    audio_chunk = None
                    if output.audio_pcm is not None:
                        audio_np = output.audio_pcm[0, 0].cpu().numpy()
                        self.audio_output.push_audio(audio_np, presence_prob)
                        audio_chunk = audio_np

                    bits = None
                    if output.bit_llr is not None:
                        raw_llr = output.bit_llr[0].cpu().numpy()
                        bits = (raw_llr > 0).astype(np.float32).ravel()

                    # ── Update shared state ────────────────────────────────
                    self._update_state({
                        "spectrum":         (freqs_mhz, psd_db),
                        "waterfall":        self._waterfall.image_normalised.copy(),
                        "constellation":    (i_pts, q_pts),
                        "signal_db":        sig_db,
                        "snr_db":           snr_db,
                        "presence_prob":    presence_prob,
                        "freq_offset":      freq_off,
                        "phase_offset":     phase_off,
                        "bits":             bits,
                        "audio_chunk":      audio_chunk,
                        "chunks_processed": self.state["chunks_processed"] + 1,
                    })

        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            logger.error(f"[ReceiverEngine] Fatal: {err_msg}\n{traceback.format_exc()}")
            self._update_state({"running": False, "error": err_msg})

    def _update_state(self, updates: dict) -> None:
        with self._lock:
            self.state.update(updates)

    def read_state(self) -> dict:
        with self._lock:
            return dict(self.state)


# ─────────────────────────────────────────────────────────────────────────────
# Gradio application
# ─────────────────────────────────────────────────────────────────────────────

class NeuralSDRApp:
    """
    Gradio-based NeuralSDR web interface.

    Parameters
    ----------
    cfg          : NeuralSDRConfig
    neural_model : NeuralReceiver
    headless     : bool — Skip audio output (useful for CI/testing).
    """

    def __init__(
        self,
        cfg:          NeuralSDRConfig,
        neural_model,
        headless: bool = False,
    ) -> None:
        if not _gradio_available:
            raise RuntimeError("Gradio is required for the GUI.  Run: pip install gradio")

        self.cfg    = cfg
        self.model  = neural_model

        self.audio  = make_audio_output(cfg.audio, headless=headless)
        self.engine = ReceiverEngine(cfg, neural_model, self.audio)
        self._ui: Optional[gr.Blocks] = None

    # ── Build UI ──────────────────────────────────────────────────────────────

    def build(self) -> "gr.Blocks":
        """Construct and return the Gradio Blocks UI."""

        theme = gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("JetBrains Mono"),
        )

        with gr.Blocks(
            title="NeuralSDR",
            theme=theme,
            css=self._custom_css(),
        ) as demo:

            # ── Header ────────────────────────────────────────────────────
            gr.HTML("""
            <div style="display:flex;align-items:center;gap:12px;padding:12px 0 4px">
              <span style="font-size:1.6rem">📡</span>
              <div>
                <h1 style="margin:0;font-size:1.3rem;color:#58a6ff">NeuralSDR</h1>
                <p style="margin:0;font-size:0.75rem;color:#8b949e">
                  100% Neural Network SDR Receiver
                </p>
              </div>
            </div>""")

            with gr.Row():

                # ── Left column: controls ──────────────────────────────────
                with gr.Column(scale=1, min_width=280):
                    self._build_control_panel()

                # ── Right column: visualizations ───────────────────────────
                with gr.Column(scale=3):
                    self._build_visualization_panel()

        self._ui = demo
        return demo

    def _build_control_panel(self):
        """Left panel: all receive controls."""

        gr.Markdown("### ⚙️ Receive Controls")

        # Status LED
        self.status_md = gr.Markdown("🔴 **Stopped**", elem_id="status")

        # Start / Stop
        with gr.Row():
            self.btn_start = gr.Button("▶ Start", variant="primary", size="sm")
            self.btn_stop  = gr.Button("⏹ Stop",  variant="stop",    size="sm")

        gr.Markdown("---")
        gr.Markdown("**📻 Signal Parameters**")

        # Frequency
        self.freq_input = gr.Number(
            value=self.cfg.center_freq / 1e6,
            label="Center Frequency (MHz)",
            precision=6,
            minimum=0.1, maximum=6000.0,
        )

        # Sample rate dropdown
        sr_choices = [label for label, _ in COMMON_SAMPLE_RATES]
        sr_default = "2.048 MS/s"
        self.sr_dropdown = gr.Dropdown(
            choices=sr_choices, value=sr_default,
            label="Sample Rate", interactive=True,
        )

        # Modulation
        mod_choices = [m.value for m in ModulationMode]
        self.mod_dropdown = gr.Dropdown(
            choices=mod_choices,
            value=self.cfg.modulation.value,
            label="Modulation Mode",
            interactive=True,
        )

        gr.Markdown("---")
        gr.Markdown("**📶 Audio Controls**")

        # Volume
        self.vol_slider = gr.Slider(
            minimum=0.0, maximum=1.0,
            value=self.cfg.audio.volume,
            step=0.01, label="Volume",
        )

        # Squelch
        self.sql_slider = gr.Slider(
            minimum=0.0, maximum=1.0,
            value=self.cfg.audio.squelch_threshold,
            step=0.01, label="Squelch Threshold",
            info="0 = always open  |  1 = always closed",
        )

        gr.Markdown("---")
        gr.Markdown("**🔌 Source**")

        source_choices = [s.value for s in SourceType]
        self.src_dropdown = gr.Dropdown(
            choices=source_choices,
            value=self.cfg.source_type.value,
            label="Input Source",
        )

        self.file_path = gr.Textbox(
            value="", label="IQ File Path",
            placeholder="/path/to/recording.cfile",
            visible=False,
        )

        gr.Markdown("---")
        gr.Markdown("**🎯 Band Presets**")
        self._build_preset_buttons()

        # Apply button
        self.apply_btn = gr.Button("✅ Apply Settings", variant="secondary", size="sm")

        # Wiring
        self.btn_start.click(self._on_start, outputs=[self.status_md])
        self.btn_stop.click(self._on_stop,   outputs=[self.status_md])
        self.apply_btn.click(
            self._on_apply_settings,
            inputs=[
                self.freq_input, self.sr_dropdown, self.mod_dropdown,
                self.vol_slider, self.sql_slider, self.src_dropdown, self.file_path,
            ],
            outputs=[self.status_md],
        )
        self.src_dropdown.change(
            lambda s: gr.update(visible=(s == "file")),
            inputs=[self.src_dropdown],
            outputs=[self.file_path],
        )

    def _build_preset_buttons(self):
        """Build grouped preset buttons from BAND_PRESETS."""
        cats = list_presets_by_category()
        for category, presets in list(cats.items())[:4]:   # top 4 categories
            with gr.Accordion(category, open=False):
                with gr.Row():
                    for preset in presets[:4]:   # max 4 per row
                        btn = gr.Button(preset.name, size="sm")
                        btn.click(
                            fn=self._make_preset_handler(preset.name),
                            outputs=[
                                self.freq_input,
                                self.mod_dropdown,
                                self.status_md,
                            ],
                        )

    def _make_preset_handler(self, preset_name: str):
        def handler():
            preset = next(
                (p for p in BAND_PRESETS.values() if p.name == preset_name), None
            )
            if preset is None:
                return gr.update(), gr.update(), "⚠️ Preset not found"
            return (
                preset.center_freq / 1e6,
                preset.modulation,
                f"🎯 Preset: **{preset.name}**",
            )
        return handler

    def _build_visualization_panel(self):
        """Right panel: all plots and info."""

        with gr.Tabs():

            # Tab 1: Spectrum + Waterfall
            with gr.TabItem("📊 Spectrum"):
                self.spectrum_plot    = gr.Plot(label="Spectrum", show_label=False)
                self.waterfall_image  = gr.Image(
                    label="Waterfall", show_label=False,
                    height=250, interactive=False,
                )

            # Tab 2: Constellation
            with gr.TabItem("🔵 Constellation"):
                self.const_plot = gr.Plot(label="Constellation", show_label=False)

            # Tab 3: Decoded bits
            with gr.TabItem("💾 Decoded Bits"):
                self.bits_plot = gr.Plot(label="Bits", show_label=False)
                self.bits_text = gr.Textbox(
                    label="Hex dump", interactive=False, max_lines=3
                )

        with gr.Row():
            with gr.Column(scale=2):
                self.signal_meter = gr.Plot(label="Signal Level", show_label=False)
            with gr.Column(scale=1):
                self.snr_bar = gr.Plot(label="SNR", show_label=False)

        self.diag_md = gr.Markdown("*Waiting for signal...*")

        # Streaming refresh (Gradio 4.x generator approach)
        self.refresh_btn = gr.Button("🔄 Refresh", size="sm", visible=False)
        self.refresh_btn.click(
            fn=self._refresh_plots,
            outputs=[
                self.spectrum_plot,
                self.waterfall_image,
                self.const_plot,
                self.signal_meter,
                self.snr_bar,
                self.bits_plot,
                self.bits_text,
                self.diag_md,
                self.status_md,
            ],
            every=0.1,   # auto-refresh every 100 ms
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_start(self):
        self.audio.open()
        self.engine.start()
        return "🟢 **Running**"

    def _on_stop(self):
        self.engine.stop()
        self.audio.close()
        return "🔴 **Stopped**"

    def _on_apply_settings(
        self, freq_mhz, sr_label, mod_str,
        volume, squelch, source_str, file_path,
    ):
        # Update config
        self.cfg.center_freq  = float(freq_mhz) * 1e6
        self.cfg.modulation   = ModulationMode(mod_str)
        self.cfg.source_type  = SourceType(source_str)

        # Resolve sample rate from label
        sr_map = {label: val for label, val in COMMON_SAMPLE_RATES}
        self.cfg.sample_rate  = float(sr_map.get(sr_label, 2_048_000))

        if source_str == "file":
            self.cfg.file_src.path = file_path

        self.cfg.audio.volume            = float(volume)
        self.cfg.audio.squelch_threshold = float(squelch)

        if self.audio.is_open:
            self.audio.set_volume(float(volume))
            self.audio.set_squelch_threshold(float(squelch))

        self.engine.update_config(self.cfg)
        return f"✅ Settings applied — {mod_str} @ {freq_mhz:.4f} MHz"

    def _refresh_plots(self):
        """Called by Gradio's `every=` timer to refresh all visual components."""
        state = self.engine.read_state()

        # Spectrum
        freqs_mhz, psd_db = state["spectrum"]
        spectrum_fig = make_spectrum_figure(
            freqs_mhz, psd_db,
            center_freq_mhz=self.cfg.center_freq / 1e6,
            height=200,
        )

        # Waterfall
        wf_img = make_waterfall_image(state["waterfall"], width=700, height=200)

        # Constellation
        i_pts, q_pts = state["constellation"]
        const_fig = make_constellation_figure(
            i_pts, q_pts,
            mod_name=self.cfg.modulation.value,
            height=320,
        )

        # Signal meter
        meter_fig = make_signal_meter_figure(
            level_db=state["signal_db"],
            squelch_open=state["presence_prob"] > self.cfg.audio.squelch_threshold,
            height=80,
        )

        # SNR bar
        snr_fig = make_snr_bar(state["snr_db"], height=80)

        # Bits
        bits_fig, bits_hex = None, ""
        if state["bits"] is not None:
            bits_fig = make_bit_display_figure(state["bits"][:128], height=80)
            byte_vals = []
            bits = state["bits"]
            for i in range(0, len(bits) - 7, 8):
                byte = int("".join(str(int(b)) for b in bits[i:i+8]), 2)
                byte_vals.append(f"{byte:02X}")
            bits_hex = " ".join(byte_vals[:32])

        # Diagnostics
        diag = format_diagnostics(
            freq_offset_norm=state["freq_offset"],
            phase_offset_rad=state["phase_offset"],
            sample_rate=self.cfg.sample_rate,
            snr_db=state["snr_db"],
            presence_prob=state["presence_prob"],
            mod_name=self.cfg.modulation.value,
            source_name=self.cfg.source_type.value,
        )

        status = "🟢 **Running**" if state["running"] else (
            f"🔴 **Error**: {state['error']}" if state["error"] else "🔴 **Stopped**"
        )

        return (
            spectrum_fig,
            wf_img,
            const_fig,
            meter_fig,
            snr_fig,
            bits_fig,
            bits_hex,
            diag,
            status,
        )

    # ── Launch ────────────────────────────────────────────────────────────────

    def launch(
        self,
        host:  str  = "127.0.0.1",
        port:  int  = 7860,
        share: bool = False,
    ) -> None:
        """Build the UI and start the Gradio server."""
        demo = self.build()
        logger.info(f"[NeuralSDRApp] Launching GUI → http://{host}:{port}")
        demo.launch(
            server_name=host,
            server_port=port,
            share=share,
            show_api=False,
            quiet=True,
        )

    # ── CSS ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _custom_css() -> str:
        return """
        body, .gradio-container { background: #0d1117 !important; }
        .dark { --background-fill-primary: #0d1117; }
        .tabitem { border-color: #30363d !important; }
        button.primary { background: #1f6feb !important; }
        button.stop    { background: #6e1c1c !important; }
        #status        { font-size: 0.9rem; padding: 4px 0; }
        .label-wrap    { color: #8b949e !important; font-size: 0.8rem; }
        """
