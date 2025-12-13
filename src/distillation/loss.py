"""Distillation loss components used to train the student."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.distillation.projection import (
    HiddenStateProjector,
    ProjectionConfig,
    ProjectionMetrics,
)


@dataclass
class DeltaDistillationConfig:
    """Configuration for the ∆-distillation loss."""

    alpha_initial: float = 0.8
    alpha_final: float = 0.5
    temperature_initial: float = 2.0
    temperature_final: float = 1.0
    delta_alpha: float = 0.05
    delta_temperature: float = 0.05
    beta: float = 0.1
    normalize_hidden: bool = True
    eps: float = 1e-8

    # Loss component flags
    use_ce: bool = True
    use_kl: bool = True
    use_frobenius: bool = True
    
    # Projection settings for feature-based alignment
    use_projection: bool = False
    projection_loss_type: str = "cosine"
    projection_layer_strategy: str = "last"
    projection_num_teacher_layers: int = 4
    projection_normalize: bool = True
    projection_dropout: float = 0.1
    projection_use_layer_norm: bool = True

    def clamp(self) -> None:
        self.alpha_initial = float(self.alpha_initial)
        self.alpha_final = float(self.alpha_final)
        self.temperature_initial = float(self.temperature_initial)
        self.temperature_final = float(self.temperature_final)
        self.delta_alpha = float(self.delta_alpha)
        self.delta_temperature = float(self.delta_temperature)
        self.beta = float(self.beta)
        if self.alpha_initial < self.alpha_final:
            self.alpha_initial = self.alpha_final
        if self.temperature_initial < self.temperature_final:
            self.temperature_initial = self.temperature_final


@dataclass
class DeltaDistillationState:
    """Mutable state that tracks schedule progress."""

    alpha_base: float
    temperature_base: float
    global_step: int = 0
    epoch: int = 0

    def snapshot(self) -> dict:
        return {
            "alpha_base": self.alpha_base,
            "temperature_base": self.temperature_base,
            "global_step": self.global_step,
            "epoch": self.epoch,
        }


@dataclass
class DistillationLossOutput:
    total: Tensor
    cross_entropy: Tensor
    kl_divergence: Tensor
    frobenius: Tensor
    alpha: float
    temperature: float
    projection_metrics: Optional[ProjectionMetrics] = None


class DeltaDistillationLoss(nn.Module):
    """Implements ∆-distillation with KL + CE + Frobenius/Projection components."""

    def __init__(
        self,
        config: DeltaDistillationConfig,
        *,
        student_dim: Optional[int] = None,
        teacher_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        config.clamp()
        self.config = config

        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="none")

        self.state = DeltaDistillationState(
            alpha_base=config.alpha_initial,
            temperature_base=config.temperature_initial,
        )
        
        # Initialize projector if enabled
        self.projector: Optional[HiddenStateProjector] = None
        self._last_projection_metrics: Optional[ProjectionMetrics] = None
        
        if config.use_projection and student_dim is not None and teacher_dim is not None:
            proj_config = ProjectionConfig(
                enabled=True,
                student_dim=student_dim,
                teacher_dim=teacher_dim,
                loss_type=config.projection_loss_type,
                layer_strategy=config.projection_layer_strategy,
                num_teacher_layers=config.projection_num_teacher_layers,
                normalize_before_loss=config.projection_normalize,
                projection_dropout=config.projection_dropout,
                use_layer_norm=config.projection_use_layer_norm,
            )
            self.projector = HiddenStateProjector(proj_config, device=device, dtype=dtype)

    def current_alpha(self) -> float:
        return self._log_schedule(
            self.state.alpha_base,
            self.config.alpha_final,
            self.state.global_step,
        )

    def current_temperature(self) -> float:
        return self._log_schedule(
            self.state.temperature_base,
            self.config.temperature_final,
            self.state.global_step,
        )

    def step(self) -> None:
        self.state.global_step += 1

    def epoch_update(self) -> None:
        self.state.alpha_base = max(
            self.state.alpha_base - self.config.delta_alpha,
            self.config.alpha_final,
        )
        self.state.temperature_base = max(
            self.state.temperature_base - self.config.delta_temperature,
            self.config.temperature_final,
        )
        self.state.epoch += 1

    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Tensor,
        *,
        student_hidden: Optional[Tensor] = None,
        teacher_hidden: Optional[Sequence[Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> DistillationLossOutput:
        """Compute the combined distillation loss."""

        if student_logits.shape != teacher_logits.shape:
            raise ValueError("Student and teacher logits must have the same shape")
        
        alpha = self.current_alpha()
        temperature = self.current_temperature()

        dtype = student_logits.dtype

        ce = torch.tensor(0.0, device=student_logits.device, dtype=torch.float)
        kl = torch.tensor(0.0, device=student_logits.device, dtype=torch.float)
        frob = torch.tensor(0.0, device=student_logits.device, dtype=torch.float)

        if self.config.use_ce:
            ce = self._cross_entropy(student_logits, labels)
        if self.config.use_kl:
            kl = self._kl_divergence(student_logits, teacher_logits, temperature)
        if self.config.use_frobenius:
            frob = self._hidden_alignment_loss(
                student_hidden,
                teacher_hidden,
                attention_mask=attention_mask,
            )

        # Compute total loss based on enabled components
        total = torch.tensor(0.0, device=student_logits.device, dtype=dtype)
        weight_sum = 0.0

        if self.config.use_ce:
            total += (1.0 - alpha - self.config.beta) * ce
            weight_sum += (1.0 - alpha - self.config.beta)
        if self.config.use_kl:
            total += alpha * kl
            weight_sum += alpha
        if self.config.use_frobenius:
            total += self.config.beta * frob
            weight_sum += self.config.beta

        # If no component is enabled, total is zero. Otherwise, normalize by sum of weights.
        if weight_sum > 0:
            total = total.to(dtype)
            total = total / weight_sum
        else:
            total = torch.tensor(0.0, device=student_logits.device, dtype=dtype)

        ce = ce.to(dtype)
        kl = kl.to(dtype)
        frob = frob.to(dtype)

        self.step()

        return DistillationLossOutput(
            total=total,
            cross_entropy=ce.detach(),
            kl_divergence=kl.detach(),
            frobenius=frob.detach(),
            alpha=alpha,
            temperature=temperature,
            projection_metrics=self._last_projection_metrics,
        )

    def _hidden_alignment_loss(
        self,
        student_hidden: Optional[Tensor],
        teacher_hidden: Optional[Sequence[Tensor]],
        *,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute hidden state alignment loss using projection or legacy Frobenius."""
        if student_hidden is None or teacher_hidden is None:
            device = student_hidden.device if student_hidden is not None else teacher_hidden[0].device  # type: ignore
            return torch.tensor(0.0, device=device)
        
        # Use projection if available
        if self.projector is not None:
            loss, metrics = self.projector(
                student_hidden,
                teacher_hidden,
                attention_mask=attention_mask,
                return_metrics=True,
            )
            self._last_projection_metrics = metrics
            return loss
        
        # Fallback to legacy Frobenius
        return self._frobenius_loss(student_hidden, teacher_hidden, attention_mask=attention_mask)

    def _cross_entropy(self, logits: Tensor, labels: Tensor) -> Tensor:
        vocab = logits.size(-1)
        view = logits.float().view(-1, vocab)
        ce = self.ce_loss(view, labels.view(-1).long())
        return ce

    def _kl_divergence(self, student_logits: Tensor, teacher_logits: Tensor, temperature: float) -> Tensor:
        if temperature <= 0:
            raise ValueError("Temperature must be positive")

        # Flatten to 2D: (batch * seq_len, vocab) so batchmean divides correctly
        batch_size, seq_len, vocab_size = student_logits.shape
        student_flat = student_logits.view(-1, vocab_size)
        teacher_flat = teacher_logits.view(-1, vocab_size)

        s_log_probs = F.log_softmax(student_flat.float() / temperature, dim=-1)
        t_probs = F.softmax(teacher_flat.float() / temperature, dim=-1)
        kl = self.kl_loss(s_log_probs, t_probs) * (temperature ** 2)
        kl = kl.sum(dim=-1).mean()  # sum over vocab, mean over batch and sequence
        return kl

    def _frobenius_loss(
        self,
        student_hidden: Optional[Tensor],
        teacher_hidden: Optional[Sequence[Tensor]],
        *,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if student_hidden is None or teacher_hidden is None:
            device = student_hidden.device if student_hidden is not None else teacher_hidden[0].device  # type: ignore[index]
            dtype = student_hidden.dtype if student_hidden is not None else teacher_hidden[0].dtype  # type: ignore[index]
            return torch.tensor(0.0, device=device, dtype=dtype)

        teacher_tensor = self._stack_teacher_hidden(
            teacher_hidden,
            device=student_hidden.device,
            dtype=student_hidden.dtype,
        )

        student_tensor = student_hidden
        if attention_mask is not None:
            mask = attention_mask.to(student_tensor.dtype).unsqueeze(-1)
            student_tensor = student_tensor * mask
            teacher_tensor = teacher_tensor * mask

        diff = (teacher_tensor - student_tensor).float()
        frob = torch.linalg.vector_norm(diff)

        if self.config.normalize_hidden:
            denom_elements = float(diff.numel())
            if attention_mask is not None:
                valid_tokens = float(attention_mask.sum().item())
                if valid_tokens > 0:
                    denom_elements = valid_tokens * diff.shape[-1]
            denom = math.sqrt(denom_elements) + self.config.eps
            frob = frob / denom

        return frob

    @staticmethod
    def _stack_teacher_hidden(
        teacher_hidden: Sequence[Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if not isinstance(teacher_hidden, (list, tuple)) or not teacher_hidden:
            raise TypeError("teacher_hidden must be a non-empty sequence of tensors")

        tensors = list(teacher_hidden)
        if len(tensors) > 1:
            tensors = tensors[1:]

        stacked = torch.stack(tensors, dim=0)
        mean_hidden = stacked.mean(dim=0)
        return mean_hidden.to(device=device, dtype=dtype)

    @staticmethod
    def _log_schedule(start: float, low: float, step: int) -> float:
        denom = 1.0 + math.log(step + 1.0)
        return float(low + (start - low) / denom)
