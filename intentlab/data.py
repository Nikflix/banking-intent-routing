"""BANKING77 acquisition and duplicate-aware development splits."""

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from intentlab.common import acquire

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
HASHES = {
    "train": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}


def normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def load_data(directory="data"):
    tables = {
        name: pd.read_csv(acquire(BASE + name + ".csv", Path(directory) / (name + ".csv"), digest))
        for name, digest in HASHES.items()
    }
    for frame in tables.values():
        if frame.isna().any().any() or set(frame.columns) != {"text", "category"}:
            raise ValueError("Unexpected BANKING77 schema")
        frame["normalized"] = frame.text.map(normalize)
    train, test = tables["train"], tables["test"]
    original = len(train)
    # Preserve the official test set. Remove overlapping/duplicate training text
    # before splitting, including conflicting labels for the same exact text.
    train = train[~train.normalized.isin(test.normalized)]
    conflicts = train.groupby("normalized").category.nunique()
    train = train[~train.normalized.isin(conflicts[conflicts > 1].index)]
    train = train.drop_duplicates("normalized").reset_index(drop=True)
    development, validation = train_test_split(
        train.index.to_numpy(), test_size=0.15, stratify=train.category, random_state=42
    )
    fit, calibration = train_test_split(
        development,
        test_size=0.15 / 0.85,
        stratify=train.loc[development, "category"],
        random_state=42,
    )
    return (
        train,
        test,
        {"train": fit, "calibration": calibration, "validation": validation},
        original - len(train),
    )
