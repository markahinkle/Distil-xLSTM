"""Run Distil-xLSTM training multiple times with different loss-component combinations."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from dataclasses import asdict
from copy import deepcopy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.distillation import DistillationTrainer
from src.models import (
    DistilXLSTMStudent,
    DistilLSTMStudent,
    load_teacher_model,
    build_student_spec_from_teacher,
    build_lstm_student_spec_from_teacher,
)
try:
    from src.models import DistilMambaStudent, build_mamba_student_spec_from_teacher  # type: ignore
except ImportError:
    DistilMambaStudent = None
    build_mamba_student_spec_from_teacher = None

# Add transformer imports
from src.models.transformer_student import (
    load_transformer_student,
    TransformerStudentResources,
    DistilQwenTransformerStudent,
    DistilVanillaTransformerStudent,
)

from src.utils import load_training_config

LOGGER = logging.getLogger("scripts.train_different_losses")


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
        raise ValueError(f"Unrecognized dtype '{name}'. Valid options: {list(mapping.keys()) + ['auto']}")
    return mapping[name]


def _read_last_metrics(metrics_path: Path):
    if not metrics_path.exists():
        return None
    last = None
    with metrics_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except Exception:
                continue
    return last


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Base directory where per-run outputs will be created.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args()

    training_config, base_loss_config = load_training_config(args.config)
    base_output = args.output_dir
    if base_output.exists():
        shutil.rmtree(base_output)
    base_output.mkdir(parents=True, exist_ok=True)

    # Define the four experiments requested
    experiments = [
        ("only_CE", dict(use_ce=True, use_kl=False, use_frobenius=False)),
        ("CE_KL", dict(use_ce=True, use_kl=True, use_frobenius=False)),
        ("CE_Fro", dict(use_ce=True, use_kl=False, use_frobenius=True)),
        ("CE_KL_Fro", dict(use_ce=True, use_kl=True, use_frobenius=True)),
    ]

    summary = []

    teacher_dtype = _resolve_dtype(training_config.teacher_dtype)
    teacher = load_teacher_model(dtype=teacher_dtype)
    tokenizer = teacher.tokenizer

    student_dtype = _resolve_dtype(training_config.student_dtype)
    student_class = training_config.student_model.lower()

    for name, flags in experiments:
        LOGGER.info("Starting experiment '%s' with flags %s", name, flags)
        run_dir = base_output / name
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Per-run copy of configs
        loss_config = deepcopy(base_loss_config)
        loss_config.use_ce = bool(flags.get("use_ce", True))
        loss_config.use_kl = bool(flags.get("use_kl", True))
        loss_config.use_frobenius = bool(flags.get("use_frobenius", False))
        
        # Projection settings
        loss_config.use_projection = bool(flags.get("use_projection", False))
        if "projection_loss_type" in flags:
            loss_config.projection_loss_type = flags["projection_loss_type"]
        if "projection_layer_strategy" in flags:
            loss_config.projection_layer_strategy = flags["projection_layer_strategy"]
        if "projection_num_teacher_layers" in flags:
            loss_config.projection_num_teacher_layers = flags["projection_num_teacher_layers"]

        train_cfg = deepcopy(training_config)
        # wire per-run artifact locations
        tensorboard_dir = run_dir / "tensorboard"
        metrics_path = run_dir / "metrics.jsonl"
        checkpoints_dir = run_dir / "checkpoints"

        train_cfg.logging.tensorboard_dir = tensorboard_dir
        train_cfg.logging.metrics_path = metrics_path
        train_cfg.checkpoint.output_dir = checkpoints_dir

        # Build student instance (mirror logic from scripts/train.py)
        spec = None
        if student_class == "xlstm":
            spec = build_student_spec_from_teacher(teacher, context_length=train_cfg.max_length)
            student = DistilXLSTMStudent.from_teacher(teacher, spec=spec, dtype=student_dtype)
        elif student_class == "lstm":
            spec = build_lstm_student_spec_from_teacher(teacher, context_length=train_cfg.max_length)
            student = DistilLSTMStudent.from_teacher(teacher, spec=spec, dtype=student_dtype)
        elif student_class == "mamba":
            if DistilMambaStudent is None:
                raise ImportError("Mamba student model not available in this environment")
            spec = build_mamba_student_spec_from_teacher(teacher, context_length=train_cfg.max_length)
            student = DistilMambaStudent.from_teacher(teacher, spec=spec, dtype=student_dtype)  # type: ignore
        elif student_class == "transformer_qwen":
            student = DistilQwenTransformerStudent.from_teacher(teacher, dtype=student_dtype)
        elif student_class == "transformer_vanilla":
            student = DistilVanillaTransformerStudent.from_teacher(
                teacher,
                dtype=student_dtype,
                max_length=train_cfg.max_length,
            )
        else:
            raise ValueError(f"Unrecognized student class '{student_class}'")

        if spec is not None:
            LOGGER.info("Using student spec: %s", spec)

        trainer = DistillationTrainer(
            teacher,
            student,
            loss_config=loss_config,
            train_config=train_cfg,
            tokenizer=tokenizer,
            output_dir=run_dir,
        )

        # Run training
        trainer.train()

        final_metrics = _read_last_metrics(metrics_path)
        summary.append(
            {
                "name": name,
                "output_dir": str(run_dir),
                "metrics_path": str(metrics_path),
                "final_metrics": final_metrics,
                "loss_config": asdict(loss_config),
            }
        )

    # Write summary file
    summary_path = base_output / "runs_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    LOGGER.info("Completed all experiments. Summary at %s", summary_path)


if __name__ == "__main__":
    main()
