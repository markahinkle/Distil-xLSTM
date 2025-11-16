"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from src.data import FineWebStreamConfig
from src.distillation import (
    CheckpointConfig,
    DeltaDistillationConfig,
    LoggingConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)


def _update_dataclass(instance, values: Dict[str, Any]):
    for key, value in values.items():
        if hasattr(instance, key):
            current = getattr(instance, key)
            if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
                _update_dataclass(current, value)
            else:
                setattr(instance, key, value)
    return instance


def load_training_config(path: Path) -> tuple[TrainingConfig, DeltaDistillationConfig]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    training = TrainingConfig()
    loss = DeltaDistillationConfig()

    if "training" in raw:
        train_raw = raw["training"]
        if "dataset" in train_raw:
            training.dataset = _update_dataclass(FineWebStreamConfig(), train_raw.pop("dataset"))
        if "optimizer" in train_raw:
            training.optimizer = _update_dataclass(OptimizerConfig(), train_raw.pop("optimizer"))
        if "scheduler" in train_raw:
            training.scheduler = _update_dataclass(SchedulerConfig(), train_raw.pop("scheduler"))
        if "checkpoint" in train_raw:
            training.checkpoint = _update_dataclass(CheckpointConfig(), train_raw.pop("checkpoint"))
            if isinstance(training.checkpoint.output_dir, str):
                training.checkpoint.output_dir = Path(training.checkpoint.output_dir)
        if "logging" in train_raw:
            training.logging = _update_dataclass(LoggingConfig(), train_raw.pop("logging"))
            if isinstance(training.logging.tensorboard_dir, str):
                training.logging.tensorboard_dir = Path(training.logging.tensorboard_dir)
        _update_dataclass(training, train_raw)

    if "loss" in raw:
        _update_dataclass(loss, raw["loss"])

    return training, loss
