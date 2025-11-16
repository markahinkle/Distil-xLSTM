"""Command-line entry point to load and test a pretrained xLSTM model."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (  # noqa: E402
    XLSTM_MODEL_ID,
    load_xlstm_teacher,
    run_xlstm_smoke_test,
)

LOGGER = logging.getLogger("scripts.test_xlstm_teacher")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        type=str,
        default=XLSTM_MODEL_ID,
        help="Hugging Face repo id for the xLSTM checkpoint.",
    )
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

    resources = load_xlstm_teacher(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
    )

    if args.skip_generation:
        LOGGER.info("xLSTM model loaded successfully; skipping generation as requested.")
        return

    output_text = run_xlstm_smoke_test(
        resources,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n=== xLSTM Smoke Test Output ===\n")
    print(output_text)
    print("\n================================")


if __name__ == "__main__":
    main()
