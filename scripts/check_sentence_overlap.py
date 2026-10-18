"""
scripts/check_sentence_overlap.py
Checks exact string matches between the 8,000-example training subset
and the 872-example validation set of SST-2.

Usage:
    python scripts/check_sentence_overlap.py
"""

import sys
import datasets

SUBSET_SIZE = 8000
SEED = 42

def main():
    # Load SST-2 via GLUE
    try:
        ds = datasets.load_dataset("glue", "sst2")
    except Exception:
        # Fallback if local HF cache or environment requires nyu-mll/glue
        ds = datasets.load_dataset("nyu-mll/glue", "sst2")

    train_split = ds["train"]
    val_split = ds["validation"]

    # Apply the exact same seeded shuffle used in src/data.py
    shuffled_train = train_split.shuffle(seed=SEED)
    subset_train = shuffled_train.select(range(SUBSET_SIZE))

    train_sentences = set(subset_train["sentence"])
    val_sentences = set(val_split["sentence"])

    # Count exact string matches
    exact_matches = train_sentences.intersection(val_sentences)
    count = len(exact_matches)

    print(f"Exact sentence matches: {count}")
    return count

if __name__ == "__main__":
    main()
