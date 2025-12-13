#!/usr/bin/env python3
"""Validation script for the hidden state projection module."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.distillation.projection import (
    HiddenStateProjector,
    ProjectionConfig,
    ProjectionLossType,
    TeacherLayerStrategy,
)
from src.distillation.loss import DeltaDistillationConfig, DeltaDistillationLoss


def test_projection_dimensions():
    """Test projection handles various dimension combinations."""
    print("\n=== Testing dimension handling ===")
    
    for student_dim, teacher_dim, desc in [(768, 768, "same"), (768, 2048, "smaller"), (2048, 768, "larger")]:
        config = ProjectionConfig(student_dim=student_dim, teacher_dim=teacher_dim, loss_type="cosine")
        projector = HiddenStateProjector(config)
        
        student = torch.randn(2, 32, student_dim)
        teacher = [torch.randn(2, 32, teacher_dim) for _ in range(12)]
        
        loss, metrics = projector(student, teacher)
        assert loss.shape == () and not torch.isnan(loss)
        print(f"  ✓ {desc}: loss={loss.item():.4f}, cos_sim={metrics.cosine_similarity:.4f}")
    
    print("  All dimension tests passed!")


def test_all_loss_types():
    """Test all supported loss types."""
    print("\n=== Testing loss types ===")
    
    student = torch.randn(2, 32, 768, requires_grad=True)
    teacher = [torch.randn(2, 32, 2048) for _ in range(12)]
    
    for loss_type in ProjectionLossType:
        config = ProjectionConfig(student_dim=768, teacher_dim=2048, loss_type=loss_type.value)
        projector = HiddenStateProjector(config)
        
        loss, _ = projector(student, teacher)
        loss.backward()
        grad_norm = student.grad.norm().item() if student.grad is not None else 0
        student.grad = None
        
        assert not torch.isnan(loss) and grad_norm > 0
        print(f"  ✓ {loss_type.value}: loss={loss.item():.4f}, grad_norm={grad_norm:.4f}")
    
    print("  All loss type tests passed!")


def test_layer_strategies():
    """Test different teacher layer selection strategies."""
    print("\n=== Testing layer strategies ===")
    
    student = torch.randn(2, 32, 768)
    teacher = [torch.randn(2, 32, 2048) for _ in range(32)]
    
    for strategy in TeacherLayerStrategy:
        config = ProjectionConfig(student_dim=768, teacher_dim=2048, layer_strategy=strategy.value, num_teacher_layers=4)
        projector = HiddenStateProjector(config)
        
        loss, _ = projector(student, teacher)
        assert not torch.isnan(loss)
        print(f"  ✓ {strategy.value}: loss={loss.item():.4f}")
    
    print("  All layer strategy tests passed!")


def test_integration():
    """Test integration with DeltaDistillationLoss."""
    print("\n=== Testing integration ===")
    
    config = DeltaDistillationConfig(use_ce=True, use_kl=True, use_frobenius=True, use_projection=True, projection_loss_type="cosine")
    loss_fn = DeltaDistillationLoss(config, student_dim=768, teacher_dim=2048)
    
    student_logits = torch.randn(2, 32, 32000, requires_grad=True)
    teacher_logits = torch.randn(2, 32, 32000)
    labels = torch.randint(0, 32000, (2, 32))
    student_hidden = torch.randn(2, 32, 768, requires_grad=True)
    teacher_hidden = [torch.randn(2, 32, 2048) for _ in range(32)]
    
    output = loss_fn(student_logits, teacher_logits, labels, student_hidden=student_hidden, teacher_hidden=teacher_hidden)
    
    assert not torch.isnan(output.total) and output.projection_metrics is not None
    output.total.backward()
    assert student_hidden.grad is not None
    
    print(f"  ✓ Total: {output.total.item():.4f}, Cos sim: {output.projection_metrics.cosine_similarity:.4f}")
    print("  Integration test passed!")


def main():
    print("=" * 50)
    print("Projection Module Validation")
    print("=" * 50)
    
    torch.manual_seed(42)
    test_projection_dimensions()
    test_all_loss_types()
    test_layer_strategies()
    test_integration()
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)


if __name__ == "__main__":
    main()
