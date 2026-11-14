"""Small runtime helpers: meters, formatting, seeding and graceful stop."""

from __future__ import annotations

import os
import random
import signal
from contextlib import suppress

import numpy as np
import torch

_STOP = {"requested": False}


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch and disable cuDNN autotuning."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def seed_worker(_worker_id: int) -> None:
    """DataLoader worker seeding function (must stay at module level for spawn)."""
    s = torch.initial_seed() % (2**32)
    np.random.seed(s)
    random.seed(s)
    with suppress(Exception):
        signal.signal(signal.SIGINT, signal.SIG_IGN)


def install_stop_handlers() -> None:
    """Turn SIGINT / SIGTERM into a flag checked at step boundaries."""

    def handler(signum, _frame):
        _STOP["requested"] = True
        name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        print(f"[sage] received {name}, stopping after the current step", flush=True)

    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(Exception):
            signal.signal(sig, handler)


def stop_requested() -> bool:
    return _STOP["requested"]


def rng_state() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_state(state: dict | None) -> bool:
    if not state:
        return False
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("python") is not None:
        random.setstate(state["python"])
    return True
