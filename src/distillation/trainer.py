"""Training pipeline for Distil-xLSTM."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import get_cosine_schedule_with_warmup

from src.data import FineWebStreamConfig, build_tokenized_dataloader, load_fineweb_stream
from src.distillation.loss import DeltaDistillationConfig, DeltaDistillationLoss
from src.models import DistilXLSTMStudent, TeacherResources


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
    dataset: FineWebStreamConfig = field(default_factory=FineWebStreamConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


class DistillationTrainer:
    """End-to-end trainer for Transformer -> xLSTM distillation."""

    def __init__(
        self,
        teacher: TeacherResources,
        student: DistilXLSTMStudent,
        *,
        loss_config: DeltaDistillationConfig,
        train_config: TrainingConfig,
        tokenizer,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.tokenizer = tokenizer
        self.device = teacher.device
        self.train_config = train_config
        self.loss_fn = DeltaDistillationLoss(loss_config)

        scaler_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler = GradScaler(scaler_device, enabled=self._use_amp)
        self.writer: Optional[SummaryWriter] = None
        if train_config.logging.tensorboard_dir is not None:
            self.writer = SummaryWriter(str(train_config.logging.tensorboard_dir))

        self.optimizer = torch.optim.AdamW(
            (p for p in student.parameters() if p.requires_grad),
            lr=train_config.optimizer.learning_rate,
            weight_decay=train_config.optimizer.weight_decay,
        )
        self.optimizer.zero_grad(set_to_none=True)

        total_steps = train_config.num_epochs * train_config.steps_per_epoch
        warmup_steps = int(total_steps * train_config.scheduler.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.enable_checkpointing = (
            train_config.checkpoint.save_every > 0 or train_config.checkpoint.keep_last > 0
        )
        self.checkpoint_dir = Path(train_config.checkpoint.output_dir)
        if self.enable_checkpointing:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0

    @property
    def _use_amp(self) -> bool:
        return self.train_config.mixed_precision and torch.cuda.is_available()

    def _build_dataloader(self):
        dataset = load_fineweb_stream(self.train_config.dataset)
        dataloader = build_tokenized_dataloader(
            dataset,
            self.tokenizer,
            batch_size=self.train_config.batch_size,
            max_length=self.train_config.max_length,
            num_workers=self.train_config.num_workers,
        )
        return dataloader

    def train(self) -> None:
        dataloader = self._build_dataloader()
        accumulation = self.train_config.gradient_accumulation_steps
        log_every = self.train_config.logging.log_every

        for epoch in range(self.train_config.num_epochs):
            for step, batch in enumerate(dataloader):
                if step >= self.train_config.steps_per_epoch:
                    break

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                loss = self._training_step(input_ids, attention_mask, labels, accumulation)

                update_step = (self.global_step + 1) % accumulation == 0

                if update_step:
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in self.student.parameters() if p.requires_grad),
                        self.train_config.max_grad_norm,
                    )
                    self._optimizer_step()

                if self.global_step % log_every == 0:
                    self._log_metrics(loss)

                if (
                    self.enable_checkpointing
                    and self.train_config.checkpoint.save_every > 0
                    and self.global_step % self.train_config.checkpoint.save_every == 0
                    and self.global_step > 0
                ):
                    self._save_checkpoint(epoch)

                self.global_step += 1

            self.loss_fn.epoch_update()
            if self.enable_checkpointing:
                self._save_checkpoint(epoch)

        if self.writer:
            self.writer.close()

    def _training_step(self, input_ids, attention_mask, labels, accumulation) -> torch.Tensor:
        self.student.train()
        self.teacher.model.eval()

        autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
        with autocast(device_type=autocast_device, enabled=self._use_amp):
            student_outputs = self.student(input_ids, return_hidden_states=True)
            with torch.no_grad():
                teacher_outputs = self.teacher.model(
                    input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )

            hidden_states = student_outputs.hidden_states[-1] if student_outputs.hidden_states else None

            distill_output = self.loss_fn(
                student_logits=student_outputs.logits,
                teacher_logits=teacher_outputs.logits,
                labels=labels,
                student_hidden=hidden_states,
                teacher_hidden=teacher_outputs.hidden_states,
                attention_mask=attention_mask,
            )

            loss = distill_output.total / accumulation

        if self._use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return distill_output.total.detach()

    def _optimizer_step(self) -> None:
        if self._use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _log_metrics(self, loss: torch.Tensor) -> None:
        metrics = {
            "loss": float(loss.cpu().item()),
            "lr": float(self.scheduler.get_last_lr()[0]),
            "alpha": self.loss_fn.current_alpha(),
            "temperature": self.loss_fn.current_temperature(),
        }
        if self.writer:
            self.writer.add_scalar("train/loss", metrics["loss"], self.global_step)
            self.writer.add_scalar("train/lr", metrics["lr"], self.global_step)
            self.writer.add_scalar("train/alpha", metrics["alpha"], self.global_step)
            self.writer.add_scalar("train/temperature", metrics["temperature"], self.global_step)
        print(json.dumps({"step": self.global_step, **metrics}))

    def _save_checkpoint(self, epoch: int) -> None:
        if not self.enable_checkpointing:
            return
        ckpt_path = self.checkpoint_dir / f"checkpoint_step{self.global_step:07d}.pt"
        payload = {
            "epoch": epoch,
            "global_step": self.global_step,
            "student_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self._use_amp else None,
            "loss_state": self.loss_fn.state.snapshot(),
            "train_config": asdict(self.train_config),
        }
        torch.save(payload, ckpt_path)
        self._prune_checkpoints()

    def _prune_checkpoints(self) -> None:
        if not self.enable_checkpointing:
            return
        keep = self.train_config.checkpoint.keep_last
        if keep <= 0:
            return
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_step*.pt"))
        for old in checkpoints[:-keep]:
            try:
                os.remove(old)
            except OSError:
                pass
