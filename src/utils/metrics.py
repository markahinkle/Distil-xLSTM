"""Utilities for logging and aggregating training metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

__all__ = ["MetricsLogger", "collect_device_memory", "to_float"]


def to_float(value: Any) -> float:
    """Convert Tensor-like or numeric objects to native floats."""

    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Unsupported value type for conversion to float: {type(value)}")


class MetricsLogger:
    """Simple JSONL metrics logger."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")

    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        record: Dict[str, Any] = {"step": step}
        for key, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                record[key] = to_float(value)
            elif isinstance(value, (int, float)):
                record[key] = float(value)
            else:
                record[key] = value
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def collect_device_memory(device: torch.device) -> Optional[float]:
    """Return allocated memory (MB) for CUDA or MPS devices."""

    if device.type == "cuda":
        mem_bytes = torch.cuda.memory_allocated(device)
    elif device.type == "mps":
        if hasattr(torch.mps, "current_allocated_memory"):
            mem_bytes = torch.mps.current_allocated_memory()
        else:
            return None
    else:
        return None

    return float(mem_bytes) / (1024.0 ** 2)

