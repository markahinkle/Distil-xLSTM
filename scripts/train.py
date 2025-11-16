"""Entry point for Distil-xLSTM training."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.distillation import DistillationTrainer
from src.models import (
    DistilXLSTMStudent,
    build_student_spec_from_teacher,
    load_teacher_model,
)
from src.utils import load_training_config

LOGGER = logging.getLogger("scripts.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "config.yaml",
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args()

    training_config, loss_config = load_training_config(args.config)
    teacher = load_teacher_model()
    tokenizer = teacher.tokenizer

    spec = build_student_spec_from_teacher(teacher, context_length=training_config.max_length)
    LOGGER.info("Using student spec: %s", spec)

    student = DistilXLSTMStudent.from_teacher(teacher, spec=spec)

    trainer = DistillationTrainer(
        teacher,
        student,
        loss_config=loss_config,
        train_config=training_config,
        tokenizer=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()
