"""Utilities for streaming and tokenizing the FineWeb dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from datasets import IterableDataset, load_dataset
from transformers import PreTrainedTokenizerBase

LOGGER = logging.getLogger(__name__)


@dataclass
class FineWebStreamConfig:
    """Configuration for streaming FineWeb."""

    dataset_name: str = "HuggingFaceFW/fineweb"
    subset: Optional[str] = "sample-10BT"
    split: str = "train"
    shuffle_buffer_size: int = 10_000
    seed: int = 42
    take: Optional[int] = None


def load_fineweb_stream(config: FineWebStreamConfig) -> IterableDataset:
    """Load the FineWeb dataset in streaming mode according to ``config``."""

    load_kwargs = {
        "split": config.split,
        "streaming": True,
    }

    if config.subset:
        dataset = load_dataset(config.dataset_name, config.subset, **load_kwargs)
    else:
        dataset = load_dataset(config.dataset_name, **load_kwargs)

    if config.shuffle_buffer_size:
        dataset = dataset.shuffle(seed=config.seed, buffer_size=config.shuffle_buffer_size)

    if config.take is not None:
        dataset = dataset.take(config.take)

    LOGGER.info(
        "Loaded FineWeb stream: dataset=%s subset=%s split=%s",
        config.dataset_name,
        config.subset,
        config.split,
    )

    return dataset


def stream_text_examples(dataset: IterableDataset, limit: Optional[int] = None) -> Iterator[str]:
    """Yield raw text strings from the streamed dataset."""

    for idx, example in enumerate(dataset):
        text = example.get("text")
        if not text:
            continue
        yield text
        if limit is not None and idx + 1 >= limit:
            break


def tokenize_texts(
    tokenizer: PreTrainedTokenizerBase,
    texts: Iterable[str],
    *,
    max_length: int = 512,
    padding: bool | str = "max_length",
) -> dict:
    """Tokenize an iterable of texts with sensible defaults for language modeling."""

    return tokenizer(
        list(texts),
        truncation=True,
        max_length=max_length,
        padding=padding,
        return_tensors="pt",
    )
