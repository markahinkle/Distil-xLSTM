"""Smoke test for the ∆-distillation loss computation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.distillation import (  # noqa: E402
    DeltaDistillationConfig,
    DeltaDistillationLoss,
)
from src.models import (  # noqa: E402
    DistilXLSTMStudent,
    build_student_spec_from_teacher,
    load_teacher_model,
)

LOGGER = logging.getLogger("scripts.test_distillation_loss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args()

    teacher = load_teacher_model()
    spec = build_student_spec_from_teacher(teacher, context_length=args.context_length)
    student = DistilXLSTMStudent.from_teacher(teacher, spec=spec)

    loss_fn = DeltaDistillationLoss(DeltaDistillationConfig())

    vocab_size = teacher.model.config.vocab_size
    device = teacher.device

    inputs = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=device)

    student_outputs = student(inputs, return_hidden_states=True)
    with torch.no_grad():
        teacher_outputs = teacher.model(
            inputs,
            output_hidden_states=True,
            use_cache=False,
        )

    labels = inputs.clone()
    hidden_states = student_outputs.hidden_states[-1] if student_outputs.hidden_states else None
    teacher_hidden = teacher_outputs.hidden_states

    if hidden_states is None:
        raise RuntimeError("Student did not return hidden states")

    distill_output = loss_fn(
        student_logits=student_outputs.logits,
        teacher_logits=teacher_outputs.logits,
        labels=labels,
        student_hidden=hidden_states,
        teacher_hidden=teacher_hidden,
    )

    LOGGER.info(
        "Loss components: total=%.4f ce=%.4f kl=%.4f frob=%.4f alpha=%.4f T=%.4f",
        distill_output.total.item(),
        distill_output.cross_entropy.item(),
        distill_output.kl_divergence.item(),
        distill_output.frobenius.item(),
        distill_output.alpha,
        distill_output.temperature,
    )

    print("\n=== Distillation Loss Summary ===")
    print(distill_output)


if __name__ == "__main__":
    main()
