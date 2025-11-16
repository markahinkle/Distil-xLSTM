"""Command-line entry point to load and test the teacher model."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import load_teacher_model, run_teacher_smoke_test  # noqa: E402

LOGGER = logging.getLogger("scripts.test_teacher")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello from Distil-xLSTM!",
        help="Prompt text used for the smoke test generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="Number of new tokens to generate during the smoke test.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Only load the model/tokenizer without running the smoke test.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for Hugging Face model files.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args()

    resources = load_teacher_model(cache_dir=args.cache_dir)

    if args.skip_generation:
        LOGGER.info("Model loaded successfully; skipping generation as requested.")
        return

    output_text = run_teacher_smoke_test(
        resources,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n=== Teacher Smoke Test Output ===\n")
    print(output_text)
    print("\n================================")


if __name__ == "__main__":
    main()
