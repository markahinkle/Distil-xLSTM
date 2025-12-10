"""Smoke test for initializing the transformer student (Qwen1.5-0.5B, zero-initialized)."""

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

from src.models.transformer_student import (
    load_transformer_student,
    run_transformer_student_smoke_test,
)

LOGGER = logging.getLogger("scripts.test_transformer_student")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2, help="Dummy batch size")
    parser.add_argument("--seq-len", type=int, default=16, help="Dummy sequence length")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args()

    student_resources = load_transformer_student()
    model = student_resources.model
    tokenizer = student_resources.tokenizer
    device = student_resources.device

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable_params / total_params if total_params else 0.0

    vocab_size = model.config.vocab_size

    dummy_input = torch.randint(
        0, vocab_size, (args.batch_size, args.seq_len), device=device
    )

    with torch.no_grad():
        output = model(dummy_input, output_hidden_states=True, use_cache=False)

    LOGGER.info("Student logits shape: %s", tuple(output.logits.shape))
    if hasattr(output, "hidden_states") and output.hidden_states is not None:
        LOGGER.info(
            "Student hidden state count=%d last shape=%s",
            len(output.hidden_states),
            tuple(output.hidden_states[-1].shape),
        )

    print("\n=== Parameter Summary ===")
    print(f"Total params   : {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Trainable ratio : {ratio:.4%}")


if __name__ == "__main__":
    main()
