"""Visualization helpers for training metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import to_float

__all__ = ["generate_report", "create_fake_metrics", "load_metrics_records"]

COLOR_SCHEME = {
    "black": "#1a1a1a",
    "red": "#d73027",
    "grey": "#636363",
    "light_grey": "#9c9c9c",
}


def load_metrics_records(path: Path) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_series(records: Iterable[Dict[str, float]], key: str) -> Tuple[List[float], List[float]]:
    steps: List[float] = []
    values: List[float] = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        steps.append(record["step"])
        values.append(value)
    return steps, values


def last_value(records: Iterable[Dict[str, float]], key: str) -> Optional[float]:
    for record in reversed(list(records)):
        value = record.get(key)
        if value is not None:
            return value
    return None


def format_float(value: Optional[float], precision: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{precision}f}"


def write_markdown_report(
    generated_paths: Dict[str, Path],
    summary: Dict[str, Optional[float]],
    output_dir: Path,
) -> Path:
    md_path = output_dir / "report.md"

    loss_caption = (
        "Tracks the evolution of the total distillation loss and its components "
        "(cross-entropy, KL divergence, Frobenius alignment)."
    )
    schedule_caption = (
        "Shows how the annealing schedule adjusts α, temperature, and learning rate during training."
    )
    diagnostics_caption = (
        "Monitors gradient norms for stability and GPU memory usage to highlight resource pressure."
    )

    lines = [
        "# Distil-xLSTM Training Report",
        "",
        "## Loss Components",
        f"*{loss_caption}*",
        f"![Loss Components]({generated_paths['loss_components'].name})",
        "",
        "## Annealing Schedule",
        f"*{schedule_caption}*",
        f"![Annealing Schedule]({generated_paths['schedule'].name})",
        "",
        "## Optimization Diagnostics",
        f"*{diagnostics_caption}*",
        f"![Optimization Diagnostics]({generated_paths['optimization_diagnostics'].name})",
        "",
        "## Summary Metrics",
        "*Final recorded values once the run completed.*",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]

    summary_labels = {
        "final_loss": "Final Loss",
        "final_cross_entropy": "Final Cross Entropy",
        "final_kl_divergence": "Final KL Divergence",
        "final_frobenius": "Final Frobenius",
        "final_learning_rate": "Final Learning Rate",
        "final_alpha": "Final Alpha",
        "final_temperature": "Final Temperature",
        "final_grad_norm": "Final Gradient Norm",
        "final_gpu_memory_mb": "Final GPU Memory (MB)",
        "num_steps": "Steps Logged",
    }

    for key, label in summary_labels.items():
        value = summary.get(key)
        if key == "num_steps":
            display = str(int(value)) if value is not None else "0"
        else:
            display = format_float(value, precision=4)
        lines.append(f"| {label} | {display} |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def generate_report(metrics_path: Path, output_dir: Path) -> Dict[str, Path]:
    records = load_metrics_records(metrics_path)
    if not records:
        raise ValueError(f"No metrics found in {metrics_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: Dict[str, Path] = {}

    # Loss components
    fig, ax = plt.subplots(figsize=(8, 4))
    for key, label, color in [
        ("loss", "Total Loss", COLOR_SCHEME["black"]),
        ("cross_entropy", "Cross Entropy", COLOR_SCHEME["red"]),
        ("kl_divergence", "KL Divergence", COLOR_SCHEME["grey"]),
        ("frobenius", "Frobenius", COLOR_SCHEME["light_grey"]),
    ]:
        xs, ys = extract_series(records, key)
        if len(xs) > 1:
            ax.plot(xs, ys, color=color, label=label, linewidth=1.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(color="#d9d9d9", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend()
    loss_path = output_dir / "loss_components.png"
    fig.savefig(loss_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    generated["loss_components"] = loss_path

    # Schedule (alpha, temperature, learning rate)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    xs_alpha, ys_alpha = extract_series(records, "alpha")
    xs_temp, ys_temp = extract_series(records, "temperature")
    if xs_alpha:
        ax1.plot(xs_alpha, ys_alpha, color=COLOR_SCHEME["red"], label="Alpha", linewidth=1.8)
    if xs_temp:
        ax1.plot(xs_temp, ys_temp, color=COLOR_SCHEME["grey"], label="Temperature", linewidth=1.8)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Alpha / Temperature")
    ax1.set_ylim(bottom=0.0)
    ax2 = ax1.twinx()
    xs_lr, ys_lr = extract_series(records, "lr")
    if xs_lr:
        ax2.plot(xs_lr, ys_lr, color=COLOR_SCHEME["black"], linestyle="--", label="Learning Rate", linewidth=1.6)
    ax1.grid(color="#d9d9d9", linestyle="--", linewidth=0.6, alpha=0.6)
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), frameon=False)
    schedule_path = output_dir / "schedule.png"
    fig.savefig(schedule_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    generated["schedule"] = schedule_path

    # Optimization diagnostics (gradient norm, memory)
    fig, ax = plt.subplots(figsize=(8, 4))
    xs_grad, ys_grad = extract_series(records, "grad_norm")
    if xs_grad:
        ax.plot(xs_grad, ys_grad, color=COLOR_SCHEME["black"], label="Grad Norm", linewidth=1.8)
        ax.set_ylabel("Gradient Norm")
    xs_mem, ys_mem = extract_series(records, "gpu_memory_mb")
    if xs_mem:
        ax2 = ax.twinx()
        ax2.plot(xs_mem, ys_mem, color=COLOR_SCHEME["red"], linestyle="--", label="GPU Memory (MB)", linewidth=1.6)
        ax2.set_ylabel("GPU Memory (MB)")
    ax.set_xlabel("Step")
    ax.grid(color="#d9d9d9", linestyle="--", linewidth=0.6, alpha=0.6)
    handles, labels = ax.get_legend_handles_labels()
    if xs_mem:
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles += handles2
        labels += labels2
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)
    diagnostics_path = output_dir / "optimization_diagnostics.png"
    fig.savefig(diagnostics_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    generated["optimization_diagnostics"] = diagnostics_path

    # Summary JSON
    summary = {
        "final_loss": last_value(records, "loss"),
        "final_cross_entropy": last_value(records, "cross_entropy"),
        "final_kl_divergence": last_value(records, "kl_divergence"),
        "final_frobenius": last_value(records, "frobenius"),
        "final_learning_rate": last_value(records, "lr"),
        "final_alpha": last_value(records, "alpha"),
        "final_temperature": last_value(records, "temperature"),
        "final_grad_norm": last_value(records, "grad_norm"),
        "final_gpu_memory_mb": last_value(records, "gpu_memory_mb"),
        "num_steps": len(records),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    generated["summary"] = summary_path

    markdown_path = write_markdown_report(generated, summary, output_dir)
    generated["markdown"] = markdown_path

    return generated


def create_fake_metrics(path: Path, num_steps: int = 50) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for step in range(num_steps):
            frac = step / max(num_steps - 1, 1)
            record = {
                "step": step,
                "loss": 2000 * math.exp(-frac * 1.5) + 100 * frac,
                "cross_entropy": 800 * math.exp(-frac * 1.2) + 50 * frac,
                "kl_divergence": 1000 * math.exp(-frac * 1.1),
                "frobenius": 400 * math.exp(-frac),
                "lr": 2e-4 * (1 - frac) + 1e-6 * frac,
                "alpha": 0.8 - 0.3 * frac,
                "temperature": 2.0 - 1.0 * frac,
                "grad_norm": 5 + 0.5 * math.sin(frac * math.pi * 2),
                "gpu_memory_mb": 1600 + 50 * frac,
            }
            fh.write(json.dumps({k: to_float(v) for k, v in record.items()}) + "\n")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, help="Path to metrics JSONL file.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for generated plots.")
    parser.add_argument("--demo", action="store_true", help="Generate demo metrics instead of reading a file.")
    parser.add_argument("--steps", type=int, default=50, help="Number of demo steps.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output
    if args.demo:
        metrics_path = output_dir / "fake_metrics.jsonl"
        create_fake_metrics(metrics_path, num_steps=args.steps)
    else:
        if args.metrics is None:
            raise ValueError("--metrics must be supplied when --demo is not set.")
        metrics_path = args.metrics

    generated = generate_report(metrics_path, output_dir)
    print(json.dumps({name: str(path) for name, path in generated.items()}, indent=2))


if __name__ == "__main__":
    main()

