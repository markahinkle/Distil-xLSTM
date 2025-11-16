"""Entry point for Distil-xLSTM training."""

from __future__ import annotations

import argparse
import logging
import sys
import shutil
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
from src.utils.report import generate_report

LOGGER = logging.getLogger("scripts.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "latest",
        help="Directory for all generated artifacts (metrics, checkpoints, reports).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate plots and markdown report in the output directory after training.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args()

    training_config, loss_config = load_training_config(args.config)
    output_dir = args.output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_dir = output_dir / "tensorboard"
    metrics_path = output_dir / "metrics.jsonl"
    checkpoints_dir = output_dir / "checkpoints"

    training_config.logging.tensorboard_dir = tensorboard_dir
    training_config.logging.metrics_path = metrics_path
    training_config.checkpoint.output_dir = checkpoints_dir
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

    if args.report:
        generated = generate_report(metrics_path, output_dir)
        LOGGER.info("Generated report artifacts: %s", {k: str(v) for k, v in generated.items()})


if __name__ == "__main__":
    main()
