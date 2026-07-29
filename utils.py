# utils.py – wspolne pomocniki: nazwy plikow i ziarna losowosci

from __future__ import annotations

import random

import numpy as np
import torch


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def set_seed(seed: int) -> None:
    """Ustawia ziarna dla random, numpy i torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """worker_init_fn dla DataLoader – powtarzalna augmentacja w workerach."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
