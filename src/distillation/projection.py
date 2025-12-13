"""Hidden state projection for cross-architecture knowledge distillation.

This module implements learned projections to align student hidden states with teacher
representations, enabling effective feature-based distillation across different architectures
(e.g., Transformer → LSTM/xLSTM/Mamba).

Research Foundation:
- FitNets (Romero et al., 2015): Introduced hint layers with linear projections
- TinyBERT (Jiao et al., 2020): MSE loss with learned projections for BERT distillation
- DistilBERT (Sanh et al., 2019): Cosine embedding loss for representation matching
- MiniLM (Wang et al., 2020): Attention transfer with dimension alignment
- PKD (Sun et al., 2019): Patient Knowledge Distillation with layer mapping
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ProjectionLossType(str, Enum):
    """Supported loss functions for hidden state alignment."""
    
    COSINE = "cosine"      # Cosine similarity - robust to magnitude differences
    MSE = "mse"            # Mean squared error - standard regression loss
    SMOOTH_L1 = "smooth_l1"  # Huber loss - robust to outliers
    CKA = "cka"            # Centered Kernel Alignment - structural similarity


class TeacherLayerStrategy(str, Enum):
    """Strategy for selecting which teacher layers to use."""
    
    LAST = "last"          # Use only the last N layers
    UNIFORM = "uniform"    # Uniformly sample N layers across all layers
    WEIGHTED = "weighted"  # Learnable weighted average of all layers


@dataclass
class ProjectionConfig:
    """Configuration for the hidden state projector."""
    
    enabled: bool = True
    student_dim: Optional[int] = None
    teacher_dim: Optional[int] = None
    loss_type: str = "cosine"
    layer_strategy: str = "last"
    num_teacher_layers: int = 4
    normalize_before_loss: bool = True
    projection_dropout: float = 0.1
    use_layer_norm: bool = True
    use_bias: bool = False


@dataclass
class ProjectionMetrics:
    """Metrics for monitoring projection effectiveness."""
    
    loss: float
    cosine_similarity: float
    student_norm: float
    teacher_norm: float
    projected_norm: float
    alignment_ratio: float
    gradient_norm: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary for logging."""
        return {
            "projection_loss": self.loss,
            "projection_cosine_sim": self.cosine_similarity,
            "projection_student_norm": self.student_norm,
            "projection_teacher_norm": self.teacher_norm,
            "projection_projected_norm": self.projected_norm,
            "projection_alignment_ratio": self.alignment_ratio,
        }


class HiddenStateProjector(nn.Module):
    """Learned projection layer for aligning student-teacher hidden states."""
    
    def __init__(
        self,
        config: ProjectionConfig,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.config = config
        
        if config.student_dim is None or config.teacher_dim is None:
            raise ValueError("student_dim and teacher_dim must be specified")
        
        self.student_dim = config.student_dim
        self.teacher_dim = config.teacher_dim
        
        self.dropout = nn.Dropout(config.projection_dropout)
        self.projection = nn.Linear(
            config.student_dim,
            config.teacher_dim,
            bias=config.use_bias,
            device=device,
            dtype=dtype,
        )
        
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.teacher_dim, device=device, dtype=dtype)
        else:
            self.layer_norm = None
        
        self.layer_weights: Optional[nn.Parameter] = None
        self._initialize_weights()
        
        try:
            self.loss_type = ProjectionLossType(config.loss_type)
        except ValueError:
            raise ValueError(f"Unknown loss_type '{config.loss_type}'")
        
        try:
            self.layer_strategy = TeacherLayerStrategy(config.layer_strategy)
        except ValueError:
            raise ValueError(f"Unknown layer_strategy '{config.layer_strategy}'")
    
    def _initialize_weights(self) -> None:
        """Initialize projection weights."""
        with torch.no_grad():
            if self.student_dim == self.teacher_dim:
                nn.init.eye_(self.projection.weight)
                self.projection.weight.add_(torch.randn_like(self.projection.weight) * 0.01)
            else:
                nn.init.xavier_uniform_(self.projection.weight, gain=0.1)
            
            if self.config.use_bias and self.projection.bias is not None:
                nn.init.zeros_(self.projection.bias)
    
    def _init_layer_weights(self, num_layers: int) -> None:
        """Initialize learnable layer weights (for WEIGHTED strategy)."""
        if self.layer_weights is None and self.layer_strategy == TeacherLayerStrategy.WEIGHTED:
            init_weights = torch.linspace(0.5, 1.5, num_layers)
            self.layer_weights = nn.Parameter(init_weights)
    
    def _select_teacher_layers(
        self,
        teacher_hidden: Sequence[Tensor],
    ) -> Tuple[Tensor, List[int]]:
        """Select and combine teacher hidden states based on strategy."""
        layers = list(teacher_hidden[1:]) if len(teacher_hidden) > 1 else list(teacher_hidden)
        num_layers = len(layers)
        
        if num_layers == 0:
            raise ValueError("No teacher hidden layers available")
        
        if self.layer_strategy == TeacherLayerStrategy.LAST:
            n = min(self.config.num_teacher_layers, num_layers)
            selected_indices = list(range(num_layers - n, num_layers))
            selected = layers[-n:]
            stacked = torch.stack(selected, dim=0)
            combined = stacked.mean(dim=0)
            
        elif self.layer_strategy == TeacherLayerStrategy.UNIFORM:
            n = min(self.config.num_teacher_layers, num_layers)
            indices = torch.linspace(0, num_layers - 1, n).long().tolist()
            selected_indices = [int(i) for i in indices]
            selected = [layers[i] for i in selected_indices]
            stacked = torch.stack(selected, dim=0)
            combined = stacked.mean(dim=0)
            
        elif self.layer_strategy == TeacherLayerStrategy.WEIGHTED:
            self._init_layer_weights(num_layers)
            selected_indices = list(range(num_layers))
            stacked = torch.stack(layers, dim=0)
            weights = F.softmax(self.layer_weights, dim=0)
            weights = weights.to(device=stacked.device, dtype=stacked.dtype)
            combined = torch.einsum("l,lbsd->bsd", weights, stacked)
        
        else:
            raise ValueError(f"Unknown layer strategy: {self.layer_strategy}")
        
        return combined, selected_indices
    
    def project(self, student_hidden: Tensor) -> Tensor:
        """Project student hidden states to teacher dimension."""
        x = self.dropout(student_hidden)
        x = self.projection(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x
    
    def _compute_loss(
        self,
        projected: Tensor,
        teacher_target: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute alignment loss."""
        if self.config.normalize_before_loss:
            projected = F.normalize(projected, p=2, dim=-1)
            teacher_target = F.normalize(teacher_target, p=2, dim=-1)
        
        if self.loss_type == ProjectionLossType.COSINE:
            return self._cosine_loss(projected, teacher_target, attention_mask)
        elif self.loss_type == ProjectionLossType.MSE:
            return self._mse_loss(projected, teacher_target, attention_mask)
        elif self.loss_type == ProjectionLossType.SMOOTH_L1:
            return self._smooth_l1_loss(projected, teacher_target, attention_mask)
        elif self.loss_type == ProjectionLossType.CKA:
            return self._cka_loss(projected, teacher_target, attention_mask)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def _cosine_loss(self, projected: Tensor, target: Tensor, mask: Optional[Tensor]) -> Tensor:
        """Cosine similarity loss: 1 - cos_sim."""
        cos_sim = F.cosine_similarity(projected, target, dim=-1)
        if mask is not None:
            mask_float = mask.to(cos_sim.dtype)
            cos_sim = cos_sim * mask_float
            loss = 1.0 - (cos_sim.sum() / (mask_float.sum() + 1e-8))
        else:
            loss = 1.0 - cos_sim.mean()
        return loss
    
    def _mse_loss(self, projected: Tensor, target: Tensor, mask: Optional[Tensor]) -> Tensor:
        """Mean squared error loss."""
        mse = (projected - target).pow(2).mean(dim=-1)
        if mask is not None:
            mask_float = mask.to(mse.dtype)
            mse = mse * mask_float
            loss = mse.sum() / (mask_float.sum() + 1e-8)
        else:
            loss = mse.mean()
        return loss
    
    def _smooth_l1_loss(self, projected: Tensor, target: Tensor, mask: Optional[Tensor]) -> Tensor:
        """Smooth L1 (Huber) loss."""
        smooth_l1 = F.smooth_l1_loss(projected, target, reduction="none")
        smooth_l1 = smooth_l1.mean(dim=-1)
        if mask is not None:
            mask_float = mask.to(smooth_l1.dtype)
            smooth_l1 = smooth_l1 * mask_float
            loss = smooth_l1.sum() / (mask_float.sum() + 1e-8)
        else:
            loss = smooth_l1.mean()
        return loss
    
    def _cka_loss(self, projected: Tensor, target: Tensor, mask: Optional[Tensor]) -> Tensor:
        """Centered Kernel Alignment loss."""
        batch_size, seq_len, dim = projected.shape
        
        if mask is not None:
            mask_flat = mask.view(-1).bool()
            X = projected.view(-1, dim)[mask_flat]
            Y = target.view(-1, dim)[mask_flat]
        else:
            X = projected.view(-1, dim)
            Y = target.view(-1, dim)
        
        n = X.size(0)
        if n < 2:
            return torch.tensor(0.0, device=projected.device, dtype=projected.dtype)
        
        X = X - X.mean(dim=0, keepdim=True)
        Y = Y - Y.mean(dim=0, keepdim=True)
        
        XtY = X.T @ Y
        XtX = X.T @ X
        YtY = Y.T @ Y
        
        hsic_xy = (XtY * XtY).sum()
        hsic_xx = (XtX * XtX).sum()
        hsic_yy = (YtY * YtY).sum()
        
        denom = torch.sqrt(hsic_xx * hsic_yy) + 1e-8
        cka = hsic_xy / denom
        
        return 1.0 - cka
    
    def _compute_metrics(
        self,
        student_hidden: Tensor,
        projected: Tensor,
        teacher_target: Tensor,
        loss: Tensor,
        attention_mask: Optional[Tensor],
    ) -> ProjectionMetrics:
        """Compute diagnostic metrics."""
        with torch.no_grad():
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(student_hidden.dtype)
                student_norm = (student_hidden * mask).norm(dim=-1).sum() / mask.sum()
                teacher_norm = (teacher_target * mask).norm(dim=-1).sum() / mask.sum()
                projected_norm = (projected * mask).norm(dim=-1).sum() / mask.sum()
            else:
                student_norm = student_hidden.norm(dim=-1).mean()
                teacher_norm = teacher_target.norm(dim=-1).mean()
                projected_norm = projected.norm(dim=-1).mean()
            
            cos_sim = F.cosine_similarity(
                F.normalize(projected, dim=-1),
                F.normalize(teacher_target, dim=-1),
                dim=-1,
            )
            if attention_mask is not None:
                mask_float = attention_mask.to(cos_sim.dtype)
                mean_cos_sim = (cos_sim * mask_float).sum() / (mask_float.sum() + 1e-8)
            else:
                mean_cos_sim = cos_sim.mean()
            
            proj_sign = (projected > 0).float()
            target_sign = (teacher_target > 0).float()
            alignment = (proj_sign == target_sign).float().mean()
        
        return ProjectionMetrics(
            loss=float(loss.item()),
            cosine_similarity=float(mean_cos_sim.item()),
            student_norm=float(student_norm.item()),
            teacher_norm=float(teacher_norm.item()),
            projected_norm=float(projected_norm.item()),
            alignment_ratio=float(alignment.item()),
        )
    
    def forward(
        self,
        student_hidden: Tensor,
        teacher_hidden: Sequence[Tensor],
        *,
        attention_mask: Optional[Tensor] = None,
        return_metrics: bool = True,
    ) -> Tuple[Tensor, Optional[ProjectionMetrics]]:
        """Compute projection loss."""
        teacher_target, _ = self._select_teacher_layers(teacher_hidden)
        teacher_target = teacher_target.to(device=student_hidden.device, dtype=student_hidden.dtype)
        
        projected = self.project(student_hidden)
        loss = self._compute_loss(projected, teacher_target, attention_mask)
        
        metrics = None
        if return_metrics:
            metrics = self._compute_metrics(student_hidden, projected, teacher_target, loss, attention_mask)
        
        return loss, metrics
