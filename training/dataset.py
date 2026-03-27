"""
training/dataset.py
===================
PyTorch Dataset and DataLoader wrappers for NeuralSDR training.

Two dataset implementations
---------------------------
1. OnlineDataset — wraps SyntheticBatchGenerator for zero-disk-I/O training.
   Generates samples on-the-fly on the GPU (or CPU).

2. H5Dataset — loads from a pre-generated HDF5 file.
   Use for reproducible validation and test sets.

The DataLoader configuration is tuned for GPU training:
  - num_workers=0 for OnlineDataset (generation is already on GPU)
  - pin_memory=True for H5Dataset (speeds up host→device transfers)
"""

from __future__ import annotations

import math
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

from ..config.settings import TrainingConfig
from .data_generator import SyntheticBatchGenerator


# ─────────────────────────────────────────────────────────────────────────────
# Online synthetic dataset (IterableDataset — infinite stream)
# ─────────────────────────────────────────────────────────────────────────────

class OnlineDataset(IterableDataset):
    """
    Infinite stream of synthetic IQ training samples.

    Wraps SyntheticBatchGenerator as a PyTorch IterableDataset.
    Yields individual samples (not batches) so the DataLoader can
    handle batching, shuffling, and multi-worker loading normally.

    Since generation is fully on-the-fly and stateless, this dataset
    has no concept of "epochs" — training should be controlled by
    step count, not epoch count.

    Parameters
    ----------
    cfg         : TrainingConfig
    chunk_size  : int
    sample_rate : float
    device      : torch.device — Generation device.
    n_per_epoch : int — Logical "epoch size" for progress tracking.
    """

    def __init__(
        self,
        cfg:         TrainingConfig,
        chunk_size:  int   = 1024,
        sample_rate: float = 2.048e6,
        device:      torch.device = torch.device("cpu"),
        n_per_epoch: int = 50_000,
    ) -> None:
        super().__init__()
        self.generator   = SyntheticBatchGenerator(
            cfg=cfg, chunk_size=chunk_size,
            sample_rate=sample_rate, device=device,
        )
        self.n_per_epoch = n_per_epoch
        self._gen_batch_size = 32   # internal generation chunk size

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield individual samples from pre-generated batches."""
        n_yielded = 0
        while n_yielded < self.n_per_epoch:
            B = min(self._gen_batch_size, self.n_per_epoch - n_yielded)
            batch = self.generator.generate_batch(B)
            for i in range(B):
                yield {k: v[i] for k, v in batch.items()}
                n_yielded += 1

    def __len__(self) -> int:
        return self.n_per_epoch


# ─────────────────────────────────────────────────────────────────────────────
# Offline H5 dataset (Map-style Dataset)
# ─────────────────────────────────────────────────────────────────────────────

class H5Dataset(Dataset):
    """
    PyTorch Dataset backed by a pre-generated HDF5 file.

    Use for validation and test sets where reproducibility is required.
    The file must be created with ``training.data_generator.write_h5_dataset()``.

    Parameters
    ----------
    path    : str — Path to .h5 file.
    split   : str — 'train' | 'val' | 'test' (used for logging only).
    """

    def __init__(self, path: str, split: str = "val") -> None:
        super().__init__()
        try:
            import h5py
        except ImportError:
            raise RuntimeError("h5py not installed.  Run: pip install h5py")

        self.path  = path
        self.split = split
        self._file = None   # lazy open (compatible with DataLoader multiprocessing)

        # Read metadata without keeping the file open
        import h5py
        with h5py.File(path, "r") as f:
            self._n      = f.attrs["n_samples"]
            self._keys   = list(f.keys())

        from loguru import logger
        logger.info(f"[H5Dataset] {split}: {self._n:,} samples from '{path}'")

    def _open(self):
        if self._file is None:
            import h5py
            self._file = h5py.File(self.path, "r", swmr=True)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._open()
        return {
            "iq":          torch.from_numpy(self._file["iq"][idx]),
            "mod_indices": torch.tensor(self._file["mod_indices"][idx], dtype=torch.int64),
            "snr_db":      torch.tensor(self._file["snr_db"][idx], dtype=torch.float32),
            "freq_offset": torch.tensor(self._file["freq_offset"][idx], dtype=torch.float32),
            "audio_gt":    torch.from_numpy(self._file["audio_gt"][idx]),
            "sym_gt":      torch.from_numpy(self._file["sym_gt"][idx]),
            "bits_gt":     torch.from_numpy(self._file["bits_gt"][idx]),
        }

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factories
# ─────────────────────────────────────────────────────────────────────────────

def make_train_loader(
    cfg:         TrainingConfig,
    chunk_size:  int   = 1024,
    sample_rate: float = 2.048e6,
    device:      torch.device = torch.device("cpu"),
) -> DataLoader:
    """
    Create the training DataLoader (online synthetic generation).

    Parameters
    ----------
    cfg         : TrainingConfig
    chunk_size  : int
    sample_rate : float
    device      : torch.device

    Returns
    -------
    DataLoader
    """
    dataset = OnlineDataset(
        cfg=cfg,
        chunk_size=chunk_size,
        sample_rate=sample_rate,
        device=device,
        n_per_epoch=cfg.samples_per_epoch,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=0,          # Online generation is already on device
        pin_memory=False,       # Tensors are already on GPU
        drop_last=True,
    )


def make_val_loader(
    h5_path:    str,
    batch_size: int = 64,
    num_workers: int = 2,
) -> DataLoader:
    """
    Create a validation DataLoader from an H5 file.

    Parameters
    ----------
    h5_path     : str
    batch_size  : int
    num_workers : int

    Returns
    -------
    DataLoader
    """
    dataset = H5Dataset(h5_path, split="val")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
