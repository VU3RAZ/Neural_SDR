"""
neural/model_registry.py
========================
Weight loading, checkpoint management, and fine-tuning utilities.

Provides a clean interface for:
  - Loading pre-trained checkpoints (full or partial)
  - Saving checkpoints with metadata
  - Freezing/unfreezing sub-modules for targeted fine-tuning
  - Adding new modulation modes at runtime (embedding expansion)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from loguru import logger

from .receiver import NeuralReceiver
from ..config.settings import ModelConfig, ModulationMode


def load_checkpoint(
    receiver: NeuralReceiver,
    path: str,
    strict: bool = False,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Load a checkpoint into a NeuralReceiver instance.

    Parameters
    ----------
    receiver : NeuralReceiver
    path     : str — Path to .pt checkpoint file.
    strict   : bool — If False, missing/extra keys are tolerated (partial load).
    device   : torch.device — Target device for loaded tensors.

    Returns
    -------
    dict  — Checkpoint metadata (epoch, loss, config, etc.)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path!r}")

    map_location = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    # Support both raw state_dict and full checkpoint dicts
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt   # assume bare state_dict

    missing, unexpected = receiver.load_state_dict(state_dict, strict=strict)

    if missing:
        logger.warning(f"[ModelRegistry] {len(missing)} missing keys (expected for partial load)")
        if len(missing) <= 10:
            for k in missing:
                logger.debug(f"  missing: {k}")

    if unexpected:
        logger.warning(f"[ModelRegistry] {len(unexpected)} unexpected keys")
        if len(unexpected) <= 10:
            for k in unexpected:
                logger.debug(f"  unexpected: {k}")

    epoch = ckpt.get("epoch", 0)
    loss  = ckpt.get("val_loss", float("nan"))
    logger.info(
        f"[ModelRegistry] Loaded checkpoint '{Path(path).name}'  "
        f"epoch={epoch}  val_loss={loss:.4f}"
    )

    return {k: v for k, v in ckpt.items() if k != "model_state_dict"}


def save_checkpoint(
    receiver: NeuralReceiver,
    path: str,
    epoch: int = 0,
    val_loss: float = float("nan"),
    optimizer_state: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> None:
    """
    Save a full training checkpoint.

    Parameters
    ----------
    receiver        : NeuralReceiver
    path            : str
    epoch           : int
    val_loss        : float
    optimizer_state : optional optimizer.state_dict()
    extra           : optional extra metadata dict
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    ckpt = {
        "model_state_dict": receiver.state_dict(),
        "epoch":            epoch,
        "val_loss":         val_loss,
        "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "param_count":      sum(p.numel() for p in receiver.parameters()),
        "d_model":          receiver.d_model,
    }
    if optimizer_state is not None:
        ckpt["optimizer_state_dict"] = optimizer_state
    if extra:
        ckpt.update(extra)

    torch.save(ckpt, path)
    size_mb = os.path.getsize(path) / 1e6
    logger.info(f"[ModelRegistry] Saved checkpoint → {path!r}  ({size_mb:.1f} MB)")


def freeze_module(module: nn.Module, name: str = "") -> None:
    """Freeze all parameters in a module (requires_grad = False)."""
    for p in module.parameters():
        p.requires_grad_(False)
    logger.info(f"[ModelRegistry] Frozen: {name or type(module).__name__}")


def unfreeze_module(module: nn.Module, name: str = "") -> None:
    """Unfreeze all parameters in a module (requires_grad = True)."""
    for p in module.parameters():
        p.requires_grad_(True)
    logger.info(f"[ModelRegistry] Unfrozen: {name or type(module).__name__}")


def setup_fine_tune(
    receiver: NeuralReceiver,
    mode: str = "heads_only",
) -> None:
    """
    Configure which parts of the model are trainable for fine-tuning.

    Modes
    -----
    'heads_only'     : Only analog + digital + squelch heads are trainable.
                       Fast, minimal catastrophic forgetting.
    'decoder_half'   : Heads + top half of transformer.
    'full'           : All parameters trainable (standard training).
    'film_only'      : Only FiLM generators + embeddings.
                       Extremely lightweight — just conditioning adaptation.
    'new_mod'        : Only modulation embedding + heads.
                       For adding a new modulation mode to a frozen backbone.

    Parameters
    ----------
    receiver : NeuralReceiver
    mode     : str
    """
    # Start by freezing everything
    freeze_module(receiver, "receiver (all)")

    if mode == "full":
        unfreeze_module(receiver, "receiver (all)")

    elif mode == "heads_only":
        unfreeze_module(receiver.analog_head, "analog_head")
        unfreeze_module(receiver.digital_head, "digital_head")
        unfreeze_module(receiver.squelch_head, "squelch_head")

    elif mode == "decoder_half":
        unfreeze_module(receiver.analog_head, "analog_head")
        unfreeze_module(receiver.digital_head, "digital_head")
        unfreeze_module(receiver.squelch_head, "squelch_head")
        n = len(receiver.transformer_layers)
        for layer in receiver.transformer_layers[n // 2:]:
            unfreeze_module(layer)
        logger.info(f"  + top {n - n//2} transformer layers")

    elif mode == "film_only":
        # Unfreeze only FiLM generators and conditioning embeddings
        unfreeze_module(receiver.conditioning, "conditioning")
        for name, module in receiver.named_modules():
            if "film" in name.lower():
                unfreeze_module(module, name)

    elif mode == "new_mod":
        unfreeze_module(receiver.conditioning.mod_embed, "mod_embedding")
        unfreeze_module(receiver.analog_head, "analog_head")
        unfreeze_module(receiver.digital_head, "digital_head")
        unfreeze_module(receiver.squelch_head, "squelch_head")

    else:
        raise ValueError(f"Unknown fine-tune mode: {mode!r}")

    trainable = sum(p.numel() for p in receiver.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in receiver.parameters())
    logger.info(
        f"[ModelRegistry] Fine-tune mode='{mode}'  "
        f"trainable={trainable:,}/{total:,} "
        f"({100*trainable/total:.1f}%)"
    )


def build_receiver(
    cfg: ModelConfig,
    device: torch.device,
    compile_model: bool = True,
) -> NeuralReceiver:
    """
    Build, initialize, optionally load weights, and optionally compile the receiver.

    Parameters
    ----------
    cfg           : ModelConfig
    device        : torch.device
    compile_model : bool — Apply torch.compile() (requires PyTorch 2.x, CUDA/CPU).

    Returns
    -------
    NeuralReceiver — ready for inference or training.
    """
    receiver = NeuralReceiver(cfg).to(device)
    receiver.print_summary()

    # Load checkpoint if specified
    if cfg.weights_path and os.path.exists(cfg.weights_path):
        load_checkpoint(receiver, cfg.weights_path, strict=cfg.strict_load, device=device)
    else:
        if cfg.weights_path:
            logger.warning(
                f"[ModelRegistry] weights_path='{cfg.weights_path}' not found — "
                f"starting with random initialization."
            )
        else:
            logger.info("[ModelRegistry] No weights_path set — random initialization.")

    # torch.compile for inference speedup (PyTorch 2.x)
    if compile_model and hasattr(torch, "compile"):
        try:
            receiver = torch.compile(receiver, mode="reduce-overhead")
            logger.info("[ModelRegistry] Model compiled with torch.compile()")
        except Exception as exc:
            logger.warning(f"[ModelRegistry] torch.compile() failed: {exc} — using eager mode")

    return receiver
