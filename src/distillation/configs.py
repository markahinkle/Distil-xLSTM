"""Configuration dataclasses for the distillation trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.data import FineWebStreamConfig

__all__ = [
    "OptimizerConfig",
    "SchedulerConfig",
    "CheckpointConfig",
    "LoggingConfig",
    "TrainingConfig",
]


@dataclass
class OptimizerConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 0.01


@dataclass
class SchedulerConfig:
    warmup_ratio: float = 0.1
    cosine_min_lr: float = 1e-6


@dataclass
class CheckpointConfig:
    output_dir: Path = Path("checkpoints")
    save_every: int = 500
    keep_last: int = 5


@dataclass
class LoggingConfig:
    log_every: int = 50
    tensorboard_dir: Optional[Path] = Path("runs/distil_xlstm")
    metrics_path: Optional[Path] = Path("artifacts/metrics.jsonl")


@dataclass
class TrainingConfig:
    num_epochs: int = 1
    steps_per_epoch: int = 100
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    mixed_precision: bool = True
    max_length: int = 512
    num_workers: int = 0
    teacher_dtype: str = "auto"
    student_dtype: str = "auto"
    student_model: str = "xlstm"
    dataset: FineWebStreamConfig = field(default_factory=FineWebStreamConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

