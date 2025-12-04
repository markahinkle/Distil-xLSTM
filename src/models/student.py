"""Student xLSTM model that distills from a Transformer teacher."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from xlstm import (
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
)

from .teacher import TeacherResources


@dataclass
class StudentArchitectureSpec:
    """High-level description used to instantiate the student stack."""

    num_blocks: int
    num_heads: int
    embedding_dim: int
    context_length: int
    dropout: float = 0.0
    conv1d_kernel_size: int = 4
    qkv_proj_blocksize: int = 4
    slstm_backend: str = "cuda"

    @property
    def slstm_positions(self) -> list[int]:
        """Return block indices that should use the sLSTM cell."""

        return list(range(0, self.num_blocks, 2))


@dataclass
class StudentForwardOutput:
    """Lightweight container mimicking the HuggingFace output dict."""

    logits: Tensor
    hidden_states: Optional[list[Tensor]] = None


def round_up_heads(num_heads: int, multiple: int = 4) -> int:
    """Round up the attention head count to a desired multiple."""

    return int(multiple * math.ceil(num_heads / multiple))


def build_student_spec_from_teacher(
    teacher: TeacherResources,
    *,
    context_length: int = 512,
    dropout: float = 0.0,
) -> StudentArchitectureSpec:
    """Derive a student architecture spec from the teacher configuration."""

    teacher_config = teacher.model.config
    num_teacher_layers = getattr(teacher_config, "num_hidden_layers", None)
    if num_teacher_layers is None:
        raise ValueError("Teacher config must define num_hidden_layers")

    num_teacher_heads = getattr(teacher_config, "num_attention_heads", None)
    if num_teacher_heads is None:
        raise ValueError("Teacher config must define num_attention_heads")

    embedding_dim = getattr(teacher_config, "hidden_size", None)
    if embedding_dim is None:
        raise ValueError("Teacher config must define hidden_size")

    num_blocks = max(1, num_teacher_layers // 2)
    num_heads = round_up_heads(num_teacher_heads)

    return StudentArchitectureSpec(
        num_blocks=num_blocks,
        num_heads=num_heads,
        embedding_dim=embedding_dim,
        context_length=context_length,
        dropout=dropout,
        slstm_backend="cuda" if teacher.device.type == "cuda" else "vanilla",
    )


def create_stack_config(spec: StudentArchitectureSpec) -> xLSTMBlockStackConfig:
    """Create an :class:`xLSTMBlockStackConfig` from the student spec."""

    mlstm_config = mLSTMLayerConfig(
        num_heads=spec.num_heads,
        conv1d_kernel_size=spec.conv1d_kernel_size,
        qkv_proj_blocksize=spec.qkv_proj_blocksize,
        dropout=spec.dropout,
    )
    mlstm_block = mLSTMBlockConfig(mlstm=mlstm_config)

    slstm_config = sLSTMLayerConfig(
        num_heads=spec.num_heads,
        conv1d_kernel_size=spec.conv1d_kernel_size,
        dropout=spec.dropout,
    )
    slstm_config.backend = spec.slstm_backend
    slstm_block = sLSTMBlockConfig(slstm=slstm_config)

    stack_config = xLSTMBlockStackConfig(
        mlstm_block=mlstm_block,
        slstm_block=slstm_block,
        num_blocks=spec.num_blocks,
        embedding_dim=spec.embedding_dim,
        context_length=spec.context_length,
        dropout=spec.dropout,
        slstm_at=spec.slstm_positions,
    )

    return stack_config


class DistilXLSTMStudent(nn.Module):
    """Student network that reuses teacher embeddings and LM head."""

    def __init__(
        self,
        stack_config: xLSTMBlockStackConfig,
        *,
        vocab_size: int,
        freeze_embeddings: bool = True,
        freeze_lm_head: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        self.stack_config = stack_config
        self.device_type = device
        self.dtype = dtype

        self.embedding = nn.Embedding(vocab_size, stack_config.embedding_dim).to(device=device, dtype=dtype)
        self.xlstm_stack = xLSTMBlockStack(stack_config).to(device=device, dtype=dtype)
        self.lm_head = nn.Linear(stack_config.embedding_dim, vocab_size, bias=False).to(device=device, dtype=dtype)

        self.embedding.weight.requires_grad = not freeze_embeddings
        self.lm_head.weight.requires_grad = not freeze_lm_head

        # Print model info
        num_blocks = stack_config.num_blocks
        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"{self.__class__.__name__}: {num_blocks} blocks, {total_params:,} parameters"
        )

    @classmethod
    def from_teacher(
        cls,
        teacher: TeacherResources,
        *,
        spec: Optional[StudentArchitectureSpec] = None,
        freeze_embeddings: bool = True,
        freeze_lm_head: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> "DistilXLSTMStudent":
        """Instantiate a student using the teacher resources for weight reuse."""

        spec = spec or build_student_spec_from_teacher(teacher)
        stack_config = create_stack_config(spec)

        device = teacher.device
        dtype = dtype or teacher.dtype

        student = cls(
            stack_config,
            vocab_size=teacher.model.config.vocab_size,
            freeze_embeddings=freeze_embeddings,
            freeze_lm_head=freeze_lm_head,
            device=device,
            dtype=dtype,
        )

        student.copy_teacher_weights(teacher)
        student.to(device=device, dtype=dtype)
        return student

    def copy_teacher_weights(self, teacher: TeacherResources) -> None:
        """Copy embedding and LM head weights from the teacher model."""

        teacher_embedding = teacher.model.get_input_embeddings()
        teacher_output = teacher.model.get_output_embeddings()

        with torch.no_grad():
            self.embedding.weight.copy_(teacher_embedding.weight.detach())
            self.lm_head.weight.copy_(teacher_output.weight.detach())

    def reset_xlstm_parameters(self) -> None:
        """Reset the trainable xLSTM block parameters."""

        self.xlstm_stack.reset_parameters()

    def forward(
        self,
        input_ids: Tensor,
        *,
        return_hidden_states: bool = False,
    ) -> StudentForwardOutput:
        """Forward pass through the student stack."""

        x = self.embedding(input_ids)
        hidden_states = [x] if return_hidden_states else None

        x = self.xlstm_stack(x)

        if return_hidden_states and hidden_states is not None:
            hidden_states.append(x)

        logits = self.lm_head(x)
        return StudentForwardOutput(logits=logits, hidden_states=hidden_states)

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        """Return the number of parameters (optionally only trainable)."""

        parameters: Iterable[Tensor]
        if trainable_only:
            parameters = (p for p in self.parameters() if p.requires_grad)
        else:
            parameters = self.parameters()
        return sum(p.numel() for p in parameters)

    def trainable_ratio(self) -> float:
        """Return the ratio of trainable to total parameters."""

        total = self.num_parameters(trainable_only=False)
        trainable = self.num_parameters(trainable_only=True)
        return float(trainable) / float(total) if total else 0.0
