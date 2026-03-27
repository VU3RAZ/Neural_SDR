#!/usr/bin/env python3
"""
run.py
======
NeuralSDR — Application Entry Point

Usage
-----
# Launch full GUI (default):
  python run.py

# Specify source and modulation:
  python run.py --source synthetic --modulation FM_WB --freq 100.0

# Use RTL-SDR hardware:
  python run.py --source rtlsdr --freq 100.0 --modulation FM_WB

# Load from IQ file:
  python run.py --source file --file-path /path/to/capture.cfile \\
                --freq 100.0 --modulation FM_WB

# Headless (no GUI, no audio — for benchmarking / pipeline testing):
  python run.py --headless --chunks 100

# Fine-tune the model on-air (experimental):
  python run.py --source synthetic --fine-tune heads_only

# Train from scratch:
  python run.py train --epochs 50 --device cuda

# Generate validation dataset:
  python run.py gen-valset --n-samples 50000

Full help:
  python run.py --help
"""

from __future__ import annotations

import os
import sys
import signal
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

# ── Add project root to sys.path (if running as a script) ────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import click
import torch
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Rich startup banner
# ─────────────────────────────────────────────────────────────────────────────

BANNER = r"""
 ███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ██╗     ███████╗██████╗ ██████╗
 ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗██║     ██╔════╝██╔══██╗██╔══██╗
 ██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║██║     ███████╗██║  ██║██████╔╝
 ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██║     ╚════██║██║  ██║██╔══██╗
 ██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║███████╗███████║██████╔╝██║  ██║
 ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═════╝ ╚═╝  ╚═╝
"""

def print_banner() -> None:
    try:
        from rich.console import Console
        from rich.panel   import Panel
        from rich.text    import Text
        console = Console()
        text = Text(BANNER, style="bold cyan")
        subtitle = Text(
            "100% Neural Network SDR Receiver  |  github.com/neuralsdr",
            style="dim white", justify="center",
        )
        console.print(Panel(text + subtitle, border_style="blue", padding=(0, 2)))
    except ImportError:
        print(BANNER)
        print("  100% Neural Network SDR Receiver")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    logger.remove()
    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | {message}"
    logger.add(sys.stderr, level=level, format=fmt, colorize=True)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        logger.add(log_file, level="DEBUG", rotation="50 MB", retention="7 days")


# ─────────────────────────────────────────────────────────────────────────────
# Device resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            d = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            d = torch.device("mps")
        else:
            d = torch.device("cpu")
    else:
        d = torch.device(device_str)

    logger.info(f"Compute device : {d}")
    if d.type == "cuda":
        logger.info(f"  GPU          : {torch.cuda.get_device_name(d)}")
        vram_gb = torch.cuda.get_device_properties(d).total_memory / 1e9
        logger.info(f"  VRAM         : {vram_gb:.1f} GB")
        logger.info(f"  PyTorch      : {torch.__version__}")
        logger.info(f"  CUDA         : {torch.version.cuda}")
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Model bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_model(cfg, device: torch.device, compile_model: bool = True):
    """Build the NeuralReceiver, load weights if available, and compile."""
    from neuralsdr.neural.model_registry import build_receiver

    logger.info("Building NeuralReceiver...")
    receiver = build_receiver(cfg.model, device, compile_model=compile_model)

    total_params = sum(p.numel() for p in receiver.parameters())
    logger.info(f"  Parameters   : {total_params:,}  ({total_params/1e6:.1f}M)")

    dtype = cfg.device.dtype
    if dtype != torch.float32:
        receiver = receiver.to(dtype)
        logger.info(f"  dtype        : {dtype}")

    return receiver


# ─────────────────────────────────────────────────────────────────────────────
# Headless pipeline mode (no GUI, no audio)
# ─────────────────────────────────────────────────────────────────────────────

def run_headless(cfg, receiver, n_chunks: int = 100) -> None:
    """
    Run the pipeline without GUI or audio — useful for benchmarking and testing.

    Parameters
    ----------
    cfg      : NeuralSDRConfig
    receiver : NeuralReceiver
    n_chunks : int — Number of IQ chunks to process.
    """
    import torch
    from neuralsdr.sources import build_source
    from neuralsdr.dsp.iq_utils import iq_to_tensor

    logger.info(f"Headless mode: processing {n_chunks} chunks")
    device = cfg.device.device

    source = build_source(cfg)
    source.open()

    receiver.eval()
    t0     = time.monotonic()
    errors = 0

    try:
        with source:
            for i in range(n_chunks):
                iq_np = source.read_samples(cfg.model.chunk_size)
                iq_t  = iq_to_tensor(iq_np, device=device).unsqueeze(0)
                mod   = cfg.modulation
                mod_idx = torch.tensor([mod.index], device=device)
                cf      = torch.tensor([cfg.center_freq], device=device)
                bw      = torch.tensor([cfg.bandwidth], device=device)

                try:
                    with torch.no_grad():
                        out = receiver(
                            iq_t, mod_idx, cf, bw,
                            mod_mode    = mod,
                            run_analog  = mod.is_analog,
                            run_digital = mod.is_digital,
                        )

                    if (i + 1) % 10 == 0:
                        snr   = out.snr_db[0].item() if out.snr_db is not None else 0
                        pres  = out.presence_prob[0].item() if out.presence_prob is not None else 0
                        logger.info(
                            f"  [{i+1:4d}/{n_chunks}] "
                            f"SNR={snr:5.1f}dB  presence={pres:.2f}"
                        )

                except Exception as exc:
                    errors += 1
                    logger.error(f"  Chunk {i} error: {exc}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    elapsed  = time.monotonic() - t0
    chunks_s = n_chunks / elapsed
    rt_factor = chunks_s * cfg.model.chunk_size / cfg.sample_rate

    logger.info("=" * 52)
    logger.info(f"Processed  : {n_chunks} chunks in {elapsed:.2f}s")
    logger.info(f"Throughput : {chunks_s:.1f} chunks/sec")
    logger.info(f"Real-time  : {rt_factor:.1f}×")
    if errors:
        logger.warning(f"Errors     : {errors}")
    logger.info("=" * 52)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.option("--source",     default="synthetic",
              type=click.Choice(["synthetic","rtlsdr","soapy","websdr","file"]),
              help="IQ acquisition source.")
@click.option("--freq",       default=100.0,  help="Center frequency (MHz).")
@click.option("--rate",       default=2.048,  help="Sample rate (MS/s).")
@click.option("--bandwidth",  default=200.0,  help="Signal bandwidth (kHz).")
@click.option("--modulation", default="FM_WB",
              type=click.Choice([m.value for m in
                                 __import__("neuralsdr.config.settings",
                                            fromlist=["ModulationMode"]).ModulationMode]),
              help="Modulation mode.")
@click.option("--file-path",  default="", help="IQ file path (--source file).")
@click.option("--checkpoint", default=None, help="Path to model weights (.pt).")
@click.option("--device",     default="auto", help="Compute device.")
@click.option("--no-compile", is_flag=True, help="Disable torch.compile().")
@click.option("--headless",   is_flag=True, help="No GUI, no audio.")
@click.option("--chunks",     default=100,  help="Chunks to process (headless mode).")
@click.option("--host",       default="127.0.0.1", help="Gradio server host.")
@click.option("--port",       default=7860,  help="Gradio server port.")
@click.option("--share",      is_flag=True,  help="Create public Gradio link.")
@click.option("--snr",        default=15.0,  help="Synthetic SNR (dB).")
@click.option("--log-level",  default="INFO",
              type=click.Choice(["DEBUG","INFO","WARNING","ERROR"]),
              help="Logging verbosity.")
@click.option("--log-file",   default="logs/neuralsdr.log", help="Log file path.")
@click.option("--fine-tune",  default=None,
              type=click.Choice(["heads_only","decoder_half","film_only","new_mod","full"]),
              help="Enable online fine-tuning (experimental).")
@click.pass_context
def main(
    ctx, source, freq, rate, bandwidth, modulation, file_path,
    checkpoint, device, no_compile, headless, chunks,
    host, port, share, snr, log_level, log_file, fine_tune,
):
    """
    NeuralSDR — 100% Neural Network SDR Receiver.

    Run without a subcommand to launch the receiver (GUI or headless).
    """
    if ctx.invoked_subcommand is not None:
        return   # delegate to subcommand

    print_banner()
    setup_logging(log_level, log_file)

    # ── Build config ──────────────────────────────────────────────────────────
    from neuralsdr.config.settings import (
        NeuralSDRConfig, ModelConfig, SourceType, ModulationMode,
        SyntheticConfig, FileSourceConfig,
    )

    mod_enum = ModulationMode(modulation)
    src_enum = SourceType(source)

    cfg = NeuralSDRConfig(
        center_freq = freq * 1e6,
        sample_rate = rate * 1e6,
        bandwidth   = bandwidth * 1e3,
        modulation  = mod_enum,
        source_type = src_enum,
    )
    cfg.model.weights_path = checkpoint
    cfg.synthetic.snr_db   = snr

    if source == "file" and file_path:
        cfg.file_src = FileSourceConfig(path=file_path)

    cfg.gui.host  = host
    cfg.gui.port  = port
    cfg.gui.share = share

    cfg.log_summary()

    # ── Build model ───────────────────────────────────────────────────────────
    dev      = resolve_device(device)
    cfg.device.device_str = str(dev)
    receiver = bootstrap_model(cfg, dev, compile_model=not no_compile)

    # ── Optional online fine-tuning ───────────────────────────────────────────
    if fine_tune:
        from neuralsdr.neural.model_registry import setup_fine_tune
        setup_fine_tune(receiver, mode=fine_tune)
        logger.info(f"Online fine-tune mode: {fine_tune}")

    # ── Graceful shutdown handler ─────────────────────────────────────────────
    _shutdown_event = threading.Event()

    def _sighandler(sig, frame):
        logger.info("\nShutdown signal received...")
        _shutdown_event.set()

    signal.signal(signal.SIGINT,  _sighandler)
    signal.signal(signal.SIGTERM, _sighandler)

    # ── Run ───────────────────────────────────────────────────────────────────
    if headless:
        run_headless(cfg, receiver, n_chunks=chunks)
    else:
        from neuralsdr.gui.app import NeuralSDRApp
        app = NeuralSDRApp(cfg, receiver, headless=False)
        try:
            app.launch(host=host, port=port, share=share)
        except Exception as exc:
            logger.error(f"GUI launch failed: {exc}")
            logger.info("Falling back to headless mode...")
            run_headless(cfg, receiver, n_chunks=chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Subcommands
# ─────────────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--epochs",      default=50,     help="Training epochs.")
@click.option("--batch-size",  default=64,     help="Batch size.")
@click.option("--lr",          default=3e-4,   help="Learning rate.")
@click.option("--checkpoint",  default=None,   help="Resume from checkpoint.")
@click.option("--device",      default="auto")
@click.option("--log-level",   default="INFO",
              type=click.Choice(["DEBUG","INFO","WARNING","ERROR"]))
def train(epochs, batch_size, lr, checkpoint, device, log_level):
    """Train the NeuralReceiver from scratch (or resume)."""
    setup_logging(log_level, "logs/train.log")
    print_banner()

    from neuralsdr.config.settings import ModelConfig, TrainingConfig
    from neuralsdr.neural.model_registry import build_receiver
    from neuralsdr.training.trainer import Trainer

    dev = resolve_device(device)
    model_cfg = ModelConfig(weights_path=checkpoint)
    train_cfg = TrainingConfig(
        epochs=epochs, batch_size=batch_size, learning_rate=lr,
    )

    receiver = build_receiver(model_cfg, dev, compile_model=False)
    trainer  = Trainer(receiver, train_cfg, dev)
    trainer.train(n_epochs=epochs)


@main.command("gen-valset")
@click.option("--n-samples", default=10_000,   help="Number of samples.")
@click.option("--output",    default="data/val.h5", help="Output H5 path.")
@click.option("--device",    default="auto")
@click.option("--log-level", default="INFO",
              type=click.Choice(["DEBUG","INFO","WARNING","ERROR"]))
def gen_valset(n_samples, output, device, log_level):
    """Pre-generate a reproducible validation dataset."""
    setup_logging(log_level)
    dev = resolve_device(device)

    from neuralsdr.training.data_generator import write_h5_dataset
    from neuralsdr.config.settings import TrainingConfig

    write_h5_dataset(path=output, n_samples=n_samples, device=dev)


@main.command()
@click.option("--device",    default="auto")
@click.option("--batch-size",default=32)
@click.option("--n-batches", default=20)
def benchmark(device, batch_size, n_batches):
    """Benchmark generator + model inference throughput."""
    setup_logging("INFO")

    dev = resolve_device(device)

    from neuralsdr.config.settings import ModelConfig, TrainingConfig
    from neuralsdr.neural.model_registry import build_receiver
    from neuralsdr.training.data_generator import SyntheticBatchGenerator

    cfg = TrainingConfig()
    gen = SyntheticBatchGenerator(cfg, chunk_size=1024, device=dev)

    logger.info("Generator benchmark...")
    sps = gen.estimate_throughput(n_batches=n_batches, batch_size=batch_size)
    logger.info(f"  {sps/1e6:.2f}M samples/sec")

    logger.info("Model benchmark...")
    model_cfg = ModelConfig()
    receiver  = build_receiver(model_cfg, dev, compile_model=False)
    receiver.eval()

    batch = gen.generate_batch(batch_size)
    t0 = time.monotonic()
    with torch.no_grad():
        for _ in range(n_batches):
            batch = gen.generate_batch(batch_size)
            receiver(
                batch["iq"], batch["mod_indices"],
                batch["center_freq"], batch["bandwidth"],
            )
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic() - t0
    logger.info(
        f"  {(n_batches * batch_size) / elapsed:.1f} inference chunks/sec  "
        f"| latency {1000 * elapsed / n_batches:.1f} ms/batch"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
