"""Utilities for streaming and tokenizing the FineWeb dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import torch
from datasets import IterableDataset, load_dataset
from torch.utils.data import DataLoader, IterableDataset as TorchIterableDataset
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


class TokenizedFineWebIterable(TorchIterableDataset):
    """Torch IterableDataset that streams tokenized FineWeb batches."""

    def __init__(
        self,
        dataset: IterableDataset,
        tokenizer: PreTrainedTokenizerBase,
        *,
        batch_size: int,
        max_length: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        import torch

        buffer: list[str] = []
        for example in self.dataset:
            text = example.get("text")
            if not text:
                continue
            buffer.append(text)
            if len(buffer) >= self.batch_size:
                tokens = self._tokenize(buffer)
                buffer.clear()
                yield tokens

        if buffer:
            yield self._tokenize(buffer)

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        import torch

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].long()
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }
        return batch


def build_tokenized_dataloader(
    dataset: IterableDataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    batch_size: int,
    max_length: int,
    num_workers: int = 0,
) -> DataLoader:
    """Wrap the streaming dataset in a PyTorch DataLoader that yields token batches."""

    iterable = TokenizedFineWebIterable(
        dataset,
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
    )

    return DataLoader(
        iterable,
        batch_size=None,
        num_workers=num_workers,
    )
