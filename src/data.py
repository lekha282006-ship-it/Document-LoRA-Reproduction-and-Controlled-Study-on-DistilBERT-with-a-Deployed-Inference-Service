"""Dataset loading, splitting, and tokenization utilities for SST-2."""

import logging
from typing import Dict, Tuple, Optional
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from datasets import load_dataset, DatasetDict

logger = logging.getLogger(__name__)


def load_raw_sst2() -> DatasetDict:
    """Loads the SST-2 dataset from Hugging Face Hub with fallback mirrors.

    Returns:
        DatasetDict containing train, validation, and test splits.
    """
    candidates = ["nyu-mll/glue", "stanfordnlp/sst2", "glue"]
    last_err = None
    for cand in candidates:
        try:
            if cand in ["glue", "nyu-mll/glue"]:
                ds = load_dataset(cand, "sst2")
            else:
                ds = load_dataset(cand)
            logger.info("Successfully loaded SST-2 via %s", cand)
            return ds
        except Exception as e:
            logger.warning("Failed loading SST-2 via %s: %s", cand, e)
            last_err = e
    raise RuntimeError(f"Unable to load SST-2 from any candidate sources: {last_err}")


class TokenizedSST2Dataset(Dataset):
    """PyTorch Dataset wrapping tokenized SST-2 examples."""

    def __init__(self, encodings: Dict[str, torch.Tensor], labels: torch.Tensor):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self) -> int:
        return len(self.labels)


def prepare_sst2_datasets(
    tokenizer: PreTrainedTokenizerBase,
    subset_size: Optional[int] = 8000,
    seed: int = 42,
    max_length: int = 128,
) -> Tuple[TokenizedSST2Dataset, TokenizedSST2Dataset, Dict[str, int]]:
    """Loads and tokenizes SST-2 splits, applying a deterministic training subset.

    Args:
        tokenizer: Hugging Face tokenizer instance.
        subset_size: Number of training examples to select (None for full set).
        seed: Random seed for deterministic subset selection.
        max_length: Maximum sequence length for padding/truncation.

    Returns:
        Tuple of (train_dataset, val_dataset, split_stats_dict).
    """
    raw_ds = load_raw_sst2()

    train_raw = raw_ds["train"]
    val_raw = raw_ds["validation"]

    stats = {
        "full_train_size": len(train_raw),
        "validation_size": len(val_raw),
    }

    if subset_size is not None and subset_size < len(train_raw):
        # Deterministically select subset
        shuffled = train_raw.shuffle(seed=seed)
        train_raw = shuffled.select(range(subset_size))

    stats["used_train_size"] = len(train_raw)

    def tokenize_split(dataset_split):
        sentences = [str(s) for s in dataset_split["sentence"]]
        encodings = tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = torch.tensor(list(dataset_split["label"]), dtype=torch.long)
        return TokenizedSST2Dataset(encodings=encodings, labels=labels)

    train_dataset = tokenize_split(train_raw)
    val_dataset = tokenize_split(val_raw)

    return train_dataset, val_dataset, stats


def get_dataloaders(
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int = 32,
    subset_size: Optional[int] = 8000,
    seed: int = 42,
    max_length: int = 128,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Builds PyTorch DataLoaders for training and validation.

    Args:
        tokenizer: Hugging Face tokenizer.
        batch_size: Batch size for DataLoader.
        subset_size: Fixed training subset size.
        seed: Random seed for shuffling.
        max_length: Maximum token length.
        num_workers: DataLoader worker count.

    Returns:
        Tuple of (train_loader, val_loader, stats).
    """
    train_ds, val_ds, stats = prepare_sst2_datasets(
        tokenizer=tokenizer,
        subset_size=subset_size,
        seed=seed,
        max_length=max_length,
    )

    # For reproducible batches across seeds
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, stats
