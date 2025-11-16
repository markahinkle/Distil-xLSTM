"""Smoke test for streaming FineWeb and tokenizing with the teacher tokenizer."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    FineWebStreamConfig,
    load_fineweb_stream,
    stream_text_examples,
    tokenize_texts,
)
from src.models import TEACHER_MODEL_ID  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

LOGGER = logging.getLogger("scripts.test_fineweb_stream")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        type=str,
        default="sample-10BT",
        help="FineWeb subset name (e.g. sample-10BT, CC-MAIN-2024-10).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of streaming samples to inspect.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max sequence length for tokenization.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for Hugging Face assets.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args()

    config = FineWebStreamConfig(subset=args.subset, take=args.limit)
    dataset = load_fineweb_stream(config)

    LOGGER.info("Loading tokenizer %s", TEACHER_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(
        TEACHER_MODEL_ID,
        use_fast=True,
        trust_remote_code=True,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts: List[str] = list(stream_text_examples(dataset, limit=args.limit))
    if not texts:
        raise RuntimeError("No text samples yielded from FineWeb stream.")

    LOGGER.info("Fetched %d text samples; first snippet: %s", len(texts), texts[0][:120].replace("\n", " "))

    tokenized = tokenize_texts(tokenizer, texts, max_length=args.max_length)
    LOGGER.info(
        "Tokenization successful: input_ids shape=%s attention_mask shape=%s",
        tuple(tokenized["input_ids"].shape),
        tuple(tokenized["attention_mask"].shape),
    )

    print("\n=== Sample Token IDs ===")
    print(tokenized["input_ids"][0][:32])


if __name__ == "__main__":
    main()
