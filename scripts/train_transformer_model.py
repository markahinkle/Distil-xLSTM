"""Entry point for training the transformer student (Qwen1.5-0.5B, randomly-initialized) via distillation."""

from __future__ import annotations

import argparse
import logging
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.distillation import DistillationTrainer
from src.models import load_teacher_model
from src.models.transformer_student import load_transformer_student
from src.utils import load_training_config
from src.utils.report import generate_report

LOGGER = logging.getLogger("scripts.train_transformer_model")


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
        default=PROJECT_ROOT / "artifacts" / "latest_transformer_student",
        help="Directory for all generated artifacts (metrics, checkpoints, reports).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate plots and markdown report in the output directory after training.",
    )
    return parser.parse_args()


def _resolve_dtype(name: str):
    name = name.lower()
    if name in ("auto", "none", "null"):
        return None
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(
            f"Unrecognized dtype '{name}'. Valid options: {list(mapping.keys()) + ['auto']}"
        )
    return mapping[name]


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
    teacher_dtype = _resolve_dtype(training_config.teacher_dtype)

    teacher = load_teacher_model(dtype=teacher_dtype)
    tokenizer = teacher.tokenizer

    # Print teacher model info
    teacher_layers = getattr(teacher.model.config, "num_hidden_layers", "N/A")
    teacher_params = sum(p.numel() for p in teacher.model.parameters())
    print(f"Teacher: {teacher_layers} layers, {teacher_params:,} parameters")

    # Load transformer student (zero-initialized)
    student_dtype = _resolve_dtype(training_config.student_dtype)
    student_resources = load_transformer_student(
        dtype=student_dtype, device=teacher.device
    )
    student = student_resources.model

    # Print student model info
    student_layers = getattr(student.config, "num_hidden_layers", "N/A")
    student_total_params = sum(p.numel() for p in student.parameters())
    student_trainable_params = sum(
        p.numel() for p in student.parameters() if p.requires_grad
    )
    print(
        f"Student: {student_layers} layers, {student_total_params:,} total parameters, {student_trainable_params:,} trainable ({100.0 * student_trainable_params / student_total_params:.2f}% trainable)"
    )

    trainer = DistillationTrainer(
        teacher,
        student,
        loss_config=loss_config,
        train_config=training_config,
        tokenizer=tokenizer,
        output_dir=output_dir,
    )

    trainer.train()

    if args.report:
        generated = generate_report(metrics_path, output_dir)
        LOGGER.info(
            "Generated report artifacts: %s", {k: str(v) for k, v in generated.items()}
        )


if __name__ == "__main__":
    main()
