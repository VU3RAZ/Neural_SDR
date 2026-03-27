"""
training/trainer.py
===================
Training loop for the Unified Neural Receiver.

Features
--------
- Mixed-precision training (torch.amp) — automatic FP16/BF16 scaling
- Cosine annealing LR schedule with linear warmup
- Gradient clipping (global norm)
- Multi-task loss logging (per-component)
- Periodic validation with metric tracking
- Automatic checkpoint saving (best val loss + every N epochs)
- TensorBoard logging
- Progress bars via tqdm
- Graceful KeyboardInterrupt handling (saves checkpoint before exiting)

Training workflow
-----------------
  trainer = Trainer(receiver, cfg, device)
  trainer.train(n_epochs=50)

Or for fine-tuning:
  from neural.model_registry import setup_fine_tune
  setup_fine_tune(receiver, mode='heads_only')
  trainer = Trainer(receiver, cfg, device, lr_override=1e-4)
  trainer.train(n_epochs=10)
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
from loguru import logger

from ..neural.receiver import NeuralReceiver, ReceiverOutput
from ..neural.model_registry import save_checkpoint
from ..config.settings import TrainingConfig, ModulationMode
from .data_generator import SyntheticBatchGenerator
from .losses import NeuralReceiverLoss, LossWeights, compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Learning rate schedule: linear warmup + cosine decay
# ─────────────────────────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(
    optimizer:     torch.optim.Optimizer,
    warmup_steps:  int,
    total_steps:   int,
    min_lr_ratio:  float = 0.05,
) -> LambdaLR:
    """
    Linear warmup followed by cosine annealing to min_lr_ratio × peak_lr.

    Parameters
    ----------
    optimizer     : AdamW optimizer
    warmup_steps  : int — Steps for linear warmup from 0 to peak LR.
    total_steps   : int — Total training steps.
    min_lr_ratio  : float — Minimum LR as a fraction of the initial LR.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics accumulator
# ─────────────────────────────────────────────────────────────────────────────

class MetricsAccumulator:
    """Rolling average tracker for loss components and metrics."""

    def __init__(self) -> None:
        self._sums:   Dict[str, float] = {}
        self._counts: Dict[str, int]   = {}

    def update(self, d: Dict[str, float]) -> None:
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if not math.isfinite(v):
                continue
            self._sums[k]   = self._sums.get(k, 0.0) + v
            self._counts[k] = self._counts.get(k, 0) + 1

    def mean(self) -> Dict[str, float]:
        return {
            k: self._sums[k] / max(1, self._counts[k])
            for k in self._sums
        }

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Trains the NeuralReceiver end-to-end.

    Parameters
    ----------
    receiver    : NeuralReceiver
    cfg         : TrainingConfig
    device      : torch.device
    lr_override : float or None — Override config learning rate (for fine-tuning).
    loss_weights: LossWeights or None
    val_h5_path : str or None — Path to pre-generated H5 validation set.
                  If None, generates validation batches on-the-fly.
    """

    def __init__(
        self,
        receiver:     NeuralReceiver,
        cfg:          TrainingConfig,
        device:       torch.device,
        lr_override:  Optional[float] = None,
        loss_weights: Optional[LossWeights] = None,
        val_h5_path:  Optional[str] = None,
    ) -> None:
        self.receiver = receiver
        self.cfg      = cfg
        self.device   = device

        # Loss function
        self.criterion = NeuralReceiverLoss(weights=loss_weights or LossWeights())

        # Optimizer — only trainable parameters
        trainable_params = [p for p in receiver.parameters() if p.requires_grad]
        lr = lr_override or cfg.learning_rate
        self.optimizer = AdamW(
            trainable_params,
            lr=lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),    # slightly more conservative β2 for stability
            eps=1e-8,
        )
        logger.info(
            f"[Trainer] AdamW: lr={lr:.2e}  "
            f"wd={cfg.weight_decay:.2e}  "
            f"params={sum(p.numel() for p in trainable_params):,}"
        )

        # Data generator (online)
        self.gen = SyntheticBatchGenerator(
            cfg=cfg,
            chunk_size=receiver.cfg.chunk_size,
            sample_rate=2.048e6,
            device=device,
        )

        # Validation
        self.val_h5_path  = val_h5_path
        self._val_gen     = SyntheticBatchGenerator(
            cfg=cfg, chunk_size=receiver.cfg.chunk_size,
            sample_rate=2.048e6, device=device,
        )

        # LR schedule (set up after knowing total steps)
        self.scheduler    = None

        # Mixed precision
        self.use_amp   = cfg.mixed_precision and device.type == "cuda"
        self.scaler    = GradScaler(enabled=self.use_amp)

        # TensorBoard
        self._tb_writer = None

        # State
        self.global_step  = 0
        self.best_val_loss = float("inf")
        self.epoch        = 0

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    def _init_tensorboard(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = os.path.join(self.cfg.checkpoint_dir, "tb_logs")
            self._tb_writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"[Trainer] TensorBoard logs → {log_dir}")
        except ImportError:
            logger.warning("[Trainer] tensorboard not installed — skipping TB logging")

    def _log_tb(self, tag: str, value: float, step: int) -> None:
        if self._tb_writer is not None:
            self._tb_writer.add_scalar(tag, value, step)

    # ── Training step ─────────────────────────────────────────────────────────

    def _train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Execute one forward + backward pass."""
        self.receiver.train()
        self.optimizer.zero_grad(set_to_none=True)

        mod_indices = batch["mod_indices"]
        # Determine which heads to run based on batch composition
        has_analog  = batch.get("is_analog",  torch.ones(1)).any().item()
        has_digital = batch.get("is_digital", torch.zeros(1)).any().item()

        # Detect dominant modulation for digital head masking
        # (use the first sample's mode — mixed batches just get unmasked logits)
        all_mods = list(ModulationMode)
        first_mod_idx = int(mod_indices[0].item())
        mod_mode = all_mods[first_mod_idx] if first_mod_idx < len(all_mods) else None

        with autocast(enabled=self.use_amp, dtype=torch.float16 if self.use_amp else torch.float32):
            output: ReceiverOutput = self.receiver(
                iq             = batch["iq"],
                mod_indices    = mod_indices,
                center_freq_hz = batch["center_freq"],
                bandwidth_hz   = batch["bandwidth"],
                mod_mode       = mod_mode if has_digital else None,
                run_analog     = bool(has_analog),
                run_digital    = bool(has_digital),
            )
            losses = self.criterion(output, batch)

        # Backward pass with gradient scaling
        self.scaler.scale(losses["total"]).backward()

        # Unscale before gradient clipping
        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            self.receiver.parameters(),
            self.cfg.grad_clip,
        )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler is not None:
            self.scheduler.step()

        return {k: float(v.item()) for k, v in losses.items()} | {
            "grad_norm": float(grad_norm.item()),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }

    # ── Validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self, n_batches: int = 50) -> Dict[str, float]:
        """Run validation over n_batches synthetic batches."""
        self.receiver.eval()
        acc = MetricsAccumulator()
        met_acc = MetricsAccumulator()

        for _ in range(n_batches):
            batch = self._val_gen.generate_batch(self.cfg.batch_size)
            all_mods = list(ModulationMode)
            first_mod = all_mods[int(batch["mod_indices"][0].item())]

            output = self.receiver(
                iq             = batch["iq"],
                mod_indices    = batch["mod_indices"],
                center_freq_hz = batch["center_freq"],
                bandwidth_hz   = batch["bandwidth"],
                mod_mode       = first_mod,
                run_analog     = True,
                run_digital    = True,
            )
            losses  = self.criterion(output, batch)
            metrics = compute_metrics(output, batch)

            acc.update({k: v.item() if isinstance(v, torch.Tensor) else v
                        for k, v in losses.items()})
            met_acc.update(metrics)

        result = acc.mean()
        result.update(met_acc.mean())
        return result

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self, n_epochs: Optional[int] = None) -> None:
        """
        Run the full training loop.

        Parameters
        ----------
        n_epochs : int or None — Number of epochs (overrides cfg.epochs if set).
        """
        n_epochs   = n_epochs or self.cfg.epochs
        steps_per_epoch = max(1, self.cfg.samples_per_epoch // self.cfg.batch_size)
        total_steps     = n_epochs * steps_per_epoch

        # Set up LR schedule
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps=self.cfg.warmup_steps,
            total_steps=total_steps,
        )

        self._init_tensorboard()

        logger.info(
            f"[Trainer] Starting training: "
            f"{n_epochs} epochs × {steps_per_epoch} steps = {total_steps:,} total steps"
        )

        try:
            for epoch in range(1, n_epochs + 1):
                self.epoch = epoch
                self._run_epoch(epoch, steps_per_epoch)

                # Validation
                logger.info(f"[Trainer] Epoch {epoch} — Running validation...")
                val_metrics = self._validate(n_batches=50)
                val_loss = val_metrics.get("total", float("inf"))

                self._log_epoch_metrics(epoch, val_metrics, prefix="val")

                logger.info(
                    f"  val_loss={val_loss:.4f}  "
                    f"ber={val_metrics.get('ber', 0):.4f}  "
                    f"squelch_acc={val_metrics.get('squelch_accuracy', 0):.3f}  "
                    f"snr_mae={val_metrics.get('snr_mae_db', 0):.2f}dB"
                )

                # Save best checkpoint
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_path = os.path.join(self.cfg.checkpoint_dir, "best.pt")
                    save_checkpoint(
                        self.receiver, best_path,
                        epoch=epoch, val_loss=val_loss,
                        optimizer_state=self.optimizer.state_dict(),
                    )
                    logger.info(f"  ★ New best: val_loss={val_loss:.4f} → {best_path}")

                # Periodic checkpoint
                if epoch % self.cfg.save_every_n_epochs == 0:
                    ckpt_path = os.path.join(
                        self.cfg.checkpoint_dir, f"epoch_{epoch:04d}.pt"
                    )
                    save_checkpoint(
                        self.receiver, ckpt_path, epoch=epoch, val_loss=val_loss
                    )

        except KeyboardInterrupt:
            logger.warning("[Trainer] Interrupted by user — saving emergency checkpoint")
            emergency_path = os.path.join(self.cfg.checkpoint_dir, "interrupted.pt")
            save_checkpoint(
                self.receiver, emergency_path,
                epoch=self.epoch, val_loss=float("nan"),
            )

        if self._tb_writer is not None:
            self._tb_writer.close()
        logger.info(
            f"[Trainer] Training complete.  "
            f"Best val_loss={self.best_val_loss:.4f}"
        )

    def _run_epoch(self, epoch: int, steps_per_epoch: int) -> None:
        """Run one training epoch."""
        try:
            from tqdm import tqdm
            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}", ncols=100)
        except ImportError:
            pbar = range(steps_per_epoch)

        step_acc = MetricsAccumulator()
        t_data = 0.0
        t_model = 0.0

        for step_i, _ in enumerate(pbar):
            # Data generation
            t0 = time.monotonic()
            batch = self.gen.generate_batch(self.cfg.batch_size)
            t_data += time.monotonic() - t0

            # Training step
            t0 = time.monotonic()
            step_metrics = self._train_step(batch)
            t_model += time.monotonic() - t0

            step_acc.update(step_metrics)
            self.global_step += 1

            # Progress bar update
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix({
                    "loss": f"{step_metrics.get('total', 0):.3f}",
                    "lr":   f"{step_metrics.get('lr', 0):.2e}",
                    "gnorm": f"{step_metrics.get('grad_norm', 0):.2f}",
                })

            # Periodic logging
            if self.global_step % self.cfg.log_every_n_steps == 0:
                avgs = step_acc.mean()
                self._log_step_metrics(avgs)
                step_acc.reset()

        if hasattr(pbar, "close"):
            pbar.close()

        logger.info(
            f"  Epoch {epoch} — "
            f"data_time={t_data:.1f}s  model_time={t_model:.1f}s"
        )

    def _log_step_metrics(self, metrics: Dict[str, float]) -> None:
        s = self.global_step
        for k, v in metrics.items():
            self._log_tb(f"train/{k}", v, s)

        logger.debug(
            f"  step={s}  "
            + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                        if k in ("total", "audio", "symbol", "bits", "freq"))
        )

    def _log_epoch_metrics(
        self,
        epoch: int,
        metrics: Dict[str, float],
        prefix: str = "val",
    ) -> None:
        for k, v in metrics.items():
            self._log_tb(f"{prefix}/{k}", v, epoch)
