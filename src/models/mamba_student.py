"""Student Mamba model that distills from a Transformer teacher."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from mamba_ssm import Mamba


from .teacher import TeacherResources


@dataclass
class StudentArchitectureSpec:
    """High-level description used to instantiate the student stack."""

    num_layers: int
    hidden_dim: int
    embedding_dim: int
    context_length: int
    state_size: int = 16
    conv_kernel: int = 4
    expand_factor: int = 2
    dropout: float = 0.0

    @property
    def intermediate_dim(self) -> int:
        """Calculate intermediate dimension for Mamba block."""
        return self.hidden_dim * self.expand_factor


@dataclass
class StudentForwardOutput:
    """Lightweight container mimicking the HuggingFace output dict."""

    logits: Tensor
    hidden_states: Optional[list[Tensor]] = None


def build_mamba_student_spec_from_teacher(
    teacher: TeacherResources,
    *,
    context_length: int = 512,
    state_size: int = 16,
    conv_kernel: int = 4,
    expand_factor: int = 2,
    dropout: float = 0.0,
) -> StudentArchitectureSpec:
    """Derive a student architecture spec from the teacher configuration."""

    teacher_config = teacher.model.config
    num_teacher_layers = getattr(teacher_config, "num_hidden_layers", None)
    if num_teacher_layers is None:
        raise ValueError("Teacher config must define num_hidden_layers")

    embedding_dim = getattr(teacher_config, "hidden_size", None)
    if embedding_dim is None:
        raise ValueError("Teacher config must define hidden_size")

    # Use half the teacher layers for Mamba
    num_layers = max(1, num_teacher_layers // 2)
    
    # Use same dimension as teacher embedding
    hidden_dim = embedding_dim

    return StudentArchitectureSpec(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        context_length=context_length,
        state_size=state_size,
        conv_kernel=conv_kernel,
        expand_factor=expand_factor,
        dropout=dropout,
    )


class MambaBlock(nn.Module):
    """Single Mamba block with layer norm and residual connection."""

    def __init__(
        self,
        dim: int,
        state_size: int = 16,
        conv_kernel: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(dim).to(device=device, dtype=dtype)
        self.mamba = Mamba(
            d_model=dim,
            d_state=state_size,
            d_conv=conv_kernel,
            expand=expand_factor,
        ).to(device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with pre-norm and residual connection."""
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class DistilMambaStudent(nn.Module):
    """Student Mamba network that reuses teacher embeddings and LM head."""

    def __init__(
        self,
        spec: StudentArchitectureSpec,
        *,
        vocab_size: int,
        freeze_embeddings: bool = True,
        freeze_lm_head: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        self.spec = spec
        self.device_type = device
        self.dtype = dtype

        self.embedding = nn.Embedding(
            vocab_size, 
            spec.embedding_dim
        ).to(device=device, dtype=dtype)
        
        self.layers = nn.ModuleList([
            MambaBlock(
                dim=spec.hidden_dim,
                state_size=spec.state_size,
                conv_kernel=spec.conv_kernel,
                expand_factor=spec.expand_factor,
                dropout=spec.dropout,
                device=device,
                dtype=dtype,
            )
            for _ in range(spec.num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(spec.hidden_dim).to(device=device, dtype=dtype)
        
        self.lm_head = nn.Linear(
            spec.hidden_dim, 
            vocab_size, 
            bias=False
        ).to(device=device, dtype=dtype)

        self.embedding.weight.requires_grad = not freeze_embeddings
        self.lm_head.weight.requires_grad = not freeze_lm_head

    @classmethod
    def from_teacher(
        cls,
        teacher: TeacherResources,
        *,
        spec: Optional[StudentArchitectureSpec] = None,
        freeze_embeddings: bool = True,
        freeze_lm_head: bool = True,
        dtype: Optional[torch.dtype] = None,
        state_size: int = 16,
        conv_kernel: int = 4,
        expand_factor: int = 2,
    ) -> "DistilMambaStudent":
        """Instantiate a student using the teacher resources for weight reuse."""

        if spec is None:
            spec = build_mamba_student_spec_from_teacher(
                teacher,
                state_size=state_size,
                conv_kernel=conv_kernel,
                expand_factor=expand_factor,
            )
        
        device = teacher.device
        dtype = dtype or teacher.dtype

        student = cls(
            spec,
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

    def reset_mamba_parameters(self) -> None:
        """Reset the trainable Mamba parameters."""

        for layer in self.layers:
            for name, param in layer.mamba.named_parameters():
                if 'weight' in name:
                    if param.dim() >= 2:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.uniform_(param, -0.1, 0.1)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        
        # Reset layer norms
        for layer in self.layers:
            nn.init.ones_(layer.norm.weight)
            nn.init.zeros_(layer.norm.bias)
        
        nn.init.ones_(self.final_norm.weight)
        nn.init.zeros_(self.final_norm.bias)

    def forward(
        self,
        input_ids: Tensor,
        *,
        return_hidden_states: bool = False,
    ) -> StudentForwardOutput:
        """Forward pass through the student Mamba."""

        x = self.embedding(input_ids)
        hidden_states = [x] if return_hidden_states else None

        # Mamba layers forward pass
        for layer in self.layers:
            x = layer(x)
            if return_hidden_states and hidden_states is not None:
                hidden_states.append(x)

        x = self.final_norm(x)

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