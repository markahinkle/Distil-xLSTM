"""Training pipeline for Distil-xLSTM."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import math

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import get_cosine_schedule_with_warmup

from src.data import FineWebStreamConfig, build_tokenized_dataloader, load_fineweb_stream
from src.distillation.configs import (
    CheckpointConfig,
    LoggingConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from src.distillation.loss import (
    DeltaDistillationConfig,
    DeltaDistillationLoss,
    DistillationLossOutput,
)
from src.models import DistilXLSTMStudent, TeacherResources
from src.utils.metrics import MetricsLogger, collect_device_memory, to_float

import logging

LOGGER = logging.getLogger(__name__)


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
        output_dir: Path,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.tokenizer = tokenizer
        self.device = teacher.device
        self.train_config = train_config
        self.loss_fn = DeltaDistillationLoss(loss_config)
        self.output_dir = Path(output_dir)

        scaler_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler = GradScaler(scaler_device, enabled=self._use_amp)
        self.writer: Optional[SummaryWriter] = None
        if train_config.logging.tensorboard_dir is not None:
            self.writer = SummaryWriter(str(train_config.logging.tensorboard_dir))

        self.metrics_logger: Optional[MetricsLogger] = None
        if train_config.logging.metrics_path is not None:
            self.metrics_logger = MetricsLogger(Path(train_config.logging.metrics_path))

        self.trainable_params = [p for p in student.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
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

        try:
            for epoch in range(self.train_config.num_epochs):
                LOGGER.info(
                    "Starting epoch %d/%d",
                    epoch + 1,
                    self.train_config.num_epochs,
                )
                for step, batch in enumerate(dataloader):
                    if step >= self.train_config.steps_per_epoch:
                        break

                    LOGGER.info(
                        "Epoch %d Step %d/%d (global step %d)",
                        epoch + 1,
                        step + 1,
                        self.train_config.steps_per_epoch,
                        self.global_step,
                    )

                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    distill_output = self._training_step(input_ids, attention_mask, labels, accumulation)

                    update_step = (self.global_step + 1) % accumulation == 0
                    grad_norm = None
                    if update_step:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.trainable_params,
                            self.train_config.max_grad_norm,
                        )
                        self._optimizer_step()

                    gpu_mem = collect_device_memory(self.device)
                    metrics_payload = self._prepare_metrics(distill_output, grad_norm, gpu_mem)

                    if self.metrics_logger:
                        self.metrics_logger.log(self.global_step, metrics_payload)

                    self._record_tensorboard(self.global_step, metrics_payload)
                    if self.global_step % log_every == 0:
                        self._print_metrics(self.global_step, metrics_payload)

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
        finally:
            if self.writer:
                self.writer.close()
            if self.metrics_logger:
                self.metrics_logger.close()

    def _training_step(
        self,
        input_ids,
        attention_mask,
        labels,
        accumulation,
    ) -> DistillationLossOutput:
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

        self._check_tensor("student_logits", student_outputs.logits)
        if hidden_states is not None:
            self._check_tensor("student_hidden", hidden_states)
        self._check_tensor("teacher_logits", teacher_outputs.logits)
        for idx, hidden in enumerate(teacher_outputs.hidden_states):
            self._check_tensor(f"teacher_hidden_{idx}", hidden)

            distill_output = self.loss_fn(
                student_logits=student_outputs.logits,
                teacher_logits=teacher_outputs.logits,
                labels=labels,
                student_hidden=hidden_states,
                teacher_hidden=teacher_outputs.hidden_states,
                attention_mask=attention_mask,
            )

            loss = distill_output.total / accumulation

        if torch.isnan(distill_output.total) or torch.isinf(distill_output.total):
            LOGGER.error(
                "Encountered non-finite loss at global step %d (loss=%s, CE=%s, KL=%s, Frobenius=%s)",
                self.global_step,
                distill_output.total.item(),
                distill_output.cross_entropy.item(),
                distill_output.kl_divergence.item(),
                distill_output.frobenius.item(),
            )
            raise FloatingPointError("Non-finite loss encountered; aborting training run.")

        if self._use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return distill_output

    def _optimizer_step(self) -> None:
        if self._use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _prepare_metrics(
        self,
        distill_output: DistillationLossOutput,
        grad_norm: Optional[torch.Tensor],
        gpu_mem_mb: Optional[float],
    ) -> dict:
        def _safe(value):
            if value is None:
                return None
            as_float = to_float(value)
            if math.isnan(as_float) or math.isinf(as_float):
                return None
            return as_float

        metrics = {
            "loss": _safe(distill_output.total),
            "cross_entropy": _safe(distill_output.cross_entropy),
            "kl_divergence": _safe(distill_output.kl_divergence),
            "frobenius": _safe(distill_output.frobenius),
            "lr": float(self.scheduler.get_last_lr()[0]),
            "alpha": float(distill_output.alpha),
            "temperature": float(distill_output.temperature),
        }
        if grad_norm is not None:
            metrics["grad_norm"] = _safe(grad_norm)
        if gpu_mem_mb is not None:
            metrics["gpu_memory_mb"] = float(gpu_mem_mb)
        return metrics

    def _record_tensorboard(self, step: int, metrics: dict) -> None:
        if not self.writer:
            return
        self.writer.add_scalar("train/loss", metrics["loss"], step)
        self.writer.add_scalar("train/cross_entropy", metrics["cross_entropy"], step)
        self.writer.add_scalar("train/kl_divergence", metrics["kl_divergence"], step)
        self.writer.add_scalar("train/frobenius", metrics["frobenius"], step)
        self.writer.add_scalar("train/lr", metrics["lr"], step)
        self.writer.add_scalar("train/alpha", metrics["alpha"], step)
        self.writer.add_scalar("train/temperature", metrics["temperature"], step)
        if "grad_norm" in metrics:
            self.writer.add_scalar("train/grad_norm", metrics["grad_norm"], step)
        if "gpu_memory_mb" in metrics:
            self.writer.add_scalar("train/gpu_memory_mb", metrics["gpu_memory_mb"], step)

    @staticmethod
    def _print_metrics(step: int, metrics: dict) -> None:
        print(json.dumps({"step": step, **metrics}))

    def _check_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if tensor is None:
            return
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            debug_dir = self.output_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            save_path = debug_dir / f"{name}_step{self.global_step}.pt"
            torch.save(tensor.detach().cpu(), save_path)
            LOGGER.error("Saved non-finite tensor '%s' to %s", name, save_path)
            raise FloatingPointError(f"{name} contains non-finite values")

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
