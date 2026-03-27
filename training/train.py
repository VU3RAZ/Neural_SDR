"""
training/train.py
=================
Command-line interface for training and fine-tuning the NeuralReceiver.

Usage
-----
# Full training from scratch:
  python -m neuralsdr.training.train train \\
    --epochs 50 --batch-size 64 --lr 3e-4 \\
    --checkpoint-dir weights/ --device auto

# Fine-tune heads only (fastest):
  python -m neuralsdr.training.train finetune \\
    --checkpoint weights/best.pt \\
    --mode heads_only --epochs 10 --lr 1e-4

# Generate a fixed validation H5 dataset:
  python -m neuralsdr.training.train gen-valset \\
    --n-samples 10000 --output data/val.h5

# Benchmark generator throughput:
  python -m neuralsdr.training.train benchmark \\
    --batch-size 64 --device auto
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on path when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import click
import torch
from loguru import logger


def _setup_logging(debug: bool) -> None:
    logger.remove()
    level = "DEBUG" if debug else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
    )
    logger.add(
        "logs/train_{time}.log",
        level="DEBUG",
        rotation="100 MB",
        retention="7 days",
    )


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


@click.group()
def cli():
    """NeuralSDR Training CLI."""
    os.makedirs("logs", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# train command
# ─────────────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--epochs",          default=50,     help="Number of training epochs.")
@click.option("--batch-size",      default=64,     help="Batch size.")
@click.option("--lr",              default=3e-4,   help="Peak learning rate.")
@click.option("--weight-decay",    default=1e-4,   help="AdamW weight decay.")
@click.option("--grad-clip",       default=1.0,    help="Gradient clipping norm.")
@click.option("--warmup-steps",    default=1000,   help="LR warmup steps.")
@click.option("--samples-epoch",   default=50_000, help="Synthetic samples per epoch.")
@click.option("--snr-min",         default=-10.0,  help="Min training SNR (dB).")
@click.option("--snr-max",         default=30.0,   help="Max training SNR (dB).")
@click.option("--chunk-size",      default=1024,   help="IQ chunk size (power of 2).")
@click.option("--checkpoint-dir",  default="weights/", help="Directory for checkpoints.")
@click.option("--checkpoint",      default=None,   help="Optional starting checkpoint.")
@click.option("--device",          default="auto", help="Device: auto|cpu|cuda|cuda:N")
@click.option("--no-compile",      is_flag=True,   help="Disable torch.compile().")
@click.option("--no-amp",          is_flag=True,   help="Disable mixed precision.")
@click.option("--debug",           is_flag=True,   help="Verbose debug logging.")
def train(
    epochs, batch_size, lr, weight_decay, grad_clip, warmup_steps,
    samples_epoch, snr_min, snr_max, chunk_size,
    checkpoint_dir, checkpoint, device, no_compile, no_amp, debug,
):
    """Train the NeuralReceiver from scratch (or resume from checkpoint)."""
    _setup_logging(debug)

    device = _resolve_device(device)
    logger.info(f"Training on device: {device}")
    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(device)}")

    from ..config.settings import ModelConfig, TrainingConfig
    from ..neural.model_registry import build_receiver
    from .trainer import Trainer

    model_cfg = ModelConfig(
        chunk_size=chunk_size,
        weights_path=checkpoint,
    )
    train_cfg = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        warmup_steps=warmup_steps,
        samples_per_epoch=samples_epoch,
        snr_range_db=(snr_min, snr_max),
        checkpoint_dir=checkpoint_dir,
        mixed_precision=not no_amp,
    )

    receiver = build_receiver(
        model_cfg, device, compile_model=not no_compile
    )

    trainer = Trainer(receiver, train_cfg, device)
    trainer.train(n_epochs=epochs)


# ─────────────────────────────────────────────────────────────────────────────
# finetune command
# ─────────────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--checkpoint",  required=True,    help="Path to pre-trained .pt checkpoint.")
@click.option("--mode",        default="heads_only",
              type=click.Choice(["heads_only", "decoder_half", "film_only", "new_mod", "full"]),
              help="Fine-tune mode (which layers to unfreeze).")
@click.option("--epochs",      default=10,       help="Fine-tune epochs.")
@click.option("--lr",          default=1e-4,     help="Fine-tune learning rate.")
@click.option("--batch-size",  default=32,       help="Batch size.")
@click.option("--snr-min",     default=0.0,      help="Min SNR (dB) — use harder SNRs for fine-tune.")
@click.option("--snr-max",     default=30.0,     help="Max SNR (dB).")
@click.option("--device",      default="auto",   help="Device.")
@click.option("--output-dir",  default="weights/finetune/", help="Output checkpoint dir.")
@click.option("--debug",       is_flag=True)
def finetune(checkpoint, mode, epochs, lr, batch_size, snr_min, snr_max,
             device, output_dir, debug):
    """Fine-tune a pre-trained NeuralReceiver."""
    _setup_logging(debug)
    device = _resolve_device(device)

    from ..config.settings import ModelConfig, TrainingConfig
    from ..neural.model_registry import build_receiver, setup_fine_tune
    from .trainer import Trainer

    model_cfg = ModelConfig(weights_path=checkpoint)
    train_cfg = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        snr_range_db=(snr_min, snr_max),
        checkpoint_dir=output_dir,
    )

    receiver = build_receiver(model_cfg, device, compile_model=False)
    setup_fine_tune(receiver, mode=mode)

    trainer = Trainer(receiver, train_cfg, device, lr_override=lr)
    trainer.train(n_epochs=epochs)


# ─────────────────────────────────────────────────────────────────────────────
# gen-valset command
# ─────────────────────────────────────────────────────────────────────────────

@cli.command("gen-valset")
@click.option("--n-samples",  default=10_000, help="Number of validation samples.")
@click.option("--output",     default="data/val.h5", help="Output H5 file path.")
@click.option("--chunk-size", default=1024,   help="IQ chunk size.")
@click.option("--device",     default="auto", help="Device for generation.")
@click.option("--debug",      is_flag=True)
def gen_valset(n_samples, output, chunk_size, device, debug):
    """Pre-generate a reproducible validation dataset (H5 format)."""
    _setup_logging(debug)
    device = _resolve_device(device)
    logger.info(f"Generating {n_samples:,} validation samples → {output!r}")

    from .data_generator import write_h5_dataset
    from ..config.settings import TrainingConfig

    write_h5_dataset(
        path=output,
        n_samples=n_samples,
        chunk_size=chunk_size,
        device=device,
        cfg=TrainingConfig(),
    )
    logger.info("Done.")


# ─────────────────────────────────────────────────────────────────────────────
# benchmark command
# ─────────────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--batch-size", default=64, help="Batch size for benchmark.")
@click.option("--n-batches",  default=20, help="Number of benchmark batches.")
@click.option("--chunk-size", default=1024)
@click.option("--device",     default="auto")
def benchmark(batch_size, n_batches, chunk_size, device):
    """Benchmark the data generator and model inference throughput."""
    device = _resolve_device(device)
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{message}")

    from ..config.settings import TrainingConfig, ModelConfig
    from .data_generator import SyntheticBatchGenerator
    from ..neural.model_registry import build_receiver
    import time

    cfg = TrainingConfig()
    gen = SyntheticBatchGenerator(cfg, chunk_size=chunk_size, device=device)

    logger.info(f"Benchmarking data generator — device={device}  batch={batch_size}")
    sps = gen.estimate_throughput(n_batches=n_batches, batch_size=batch_size)
    logger.info(f"  Generator: {sps/1e6:.2f}M samples/sec  ({batch_size*chunk_size*n_batches:,} total)")

    # Model inference benchmark
    logger.info("Benchmarking model inference...")
    model_cfg = ModelConfig(chunk_size=chunk_size)
    receiver = build_receiver(model_cfg, device, compile_model=False)
    receiver.eval()

    batch = gen.generate_batch(batch_size)
    all_mods = list(__import__("neuralsdr.config.settings", fromlist=["ModulationMode"]).ModulationMode)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            receiver(
                batch["iq"], batch["mod_indices"],
                batch["center_freq"], batch["bandwidth"],
                run_analog=True, run_digital=True,
            )

    t0 = time.monotonic()
    with torch.no_grad():
        for _ in range(n_batches):
            batch = gen.generate_batch(batch_size)
            receiver(
                batch["iq"], batch["mod_indices"],
                batch["center_freq"], batch["bandwidth"],
                run_analog=True, run_digital=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic() - t0

    samples_sec = (n_batches * batch_size) / elapsed
    chunks_sec  = samples_sec
    realtime_factor = samples_sec * chunk_size / 2.048e6  # relative to 2 MS/s

    logger.info(f"  Inference: {chunks_sec:.1f} chunks/sec  "
                f"({realtime_factor:.1f}× real-time at 2 MS/s)")
    logger.info(f"  Latency per batch: {1000*elapsed/n_batches:.1f} ms")


if __name__ == "__main__":
    cli()
