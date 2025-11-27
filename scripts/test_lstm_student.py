"""Smoke test for initializing the student LSTM from the transformer teacher."""
from __future__ import annotations
import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (  # noqa: E402
    DistilLSTMStudent,
    build_lstm_student_spec_from_teacher,
    load_teacher_model,
)

LOGGER = logging.getLogger("scripts.test_student")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2, help="Dummy batch size")
    parser.add_argument("--seq-len", type=int, default=16, help="Dummy sequence length")
    parser.add_argument("--context-length", type=int, default=512, help="Student context length")
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional LSTM")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args()

    teacher = load_teacher_model()
    spec = build_lstm_student_spec_from_teacher(
        teacher, 
        context_length=args.context_length,
        bidirectional=args.bidirectional,
    )
    LOGGER.info("Derived student spec: %s", asdict(spec))

    student = DistilLSTMStudent.from_teacher(
        teacher, 
        spec=spec,
        bidirectional=args.bidirectional,
    )

    total_params = student.num_parameters()
    trainable_params = student.num_parameters(trainable_only=True)
    ratio = student.trainable_ratio()
    LOGGER.info(
        "Student parameters: total=%d trainable=%d ratio=%.4f",
        total_params,
        trainable_params,
        ratio,
    )

    vocab_size = teacher.model.config.vocab_size
    device = teacher.device
    dummy_input = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=device)

    with torch.no_grad():
        student_output = student(dummy_input, return_hidden_states=True)
        teacher_output = teacher.model(
            dummy_input,
            output_hidden_states=True,
            use_cache=False,
        )

    LOGGER.info("Student logits shape: %s", tuple(student_output.logits.shape))
    if student_output.hidden_states:
        LOGGER.info(
            "Student hidden state count=%d last shape=%s",
            len(student_output.hidden_states),
            tuple(student_output.hidden_states[-1].shape),
        )

    LOGGER.info(
        "Teacher hidden states=%d last shape=%s",
        len(teacher_output.hidden_states),
        tuple(teacher_output.hidden_states[-1].shape),
    )

    print("\n=== Parameter Summary ===")
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Trainable ratio : {ratio:.4%}")
    print(f"Bidirectional   : {args.bidirectional}")
    print(f"LSTM layers     : {spec.num_layers}")
    print(f"Hidden dim      : {spec.hidden_dim}")


if __name__ == "__main__":
    main()