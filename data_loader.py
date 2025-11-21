"""
Data loading utilities for GameTox and Reddit usernames datasets.
"""

from enum import IntEnum
from pathlib import Path
import pandas as pd
import numpy as np


class ToxicityLabel(IntEnum):
    """GameTox toxicity classification labels."""

    NON_TOXIC = 0
    INSULTS_AND_FLAMING = 1
    OTHER_OFFENSIVE = 2
    HATE_AND_HARASSMENT = 3
    THREATS = 4
    EXTREMISM = 5


def load_gametox(data_path: str = "data/GameTox/gametox.csv") -> pd.DataFrame:
    """Load the GameTox dataset.

    Args:
        data_path: Path to gametox.csv file

    Returns:
        DataFrame with columns:
            - message: The chat message text
            - label: Toxicity category (0-5, see ToxicityLabel enum)

    Raises:
        FileNotFoundError: If the data file doesn't exist
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"GameTox dataset not found at {data_path}. "
            "Run 'cd data && bash download.sh' to download it."
        )

    df = pd.read_csv(path)

    df = df.dropna()

    # Convert labels to integers for consistency
    df["label"] = df["label"].astype(int)

    return df


def is_toxic(label: int) -> bool:
    """Check if a label represents toxic content.

    Args:
        label: Toxicity category from ToxicityLabel enum

    Returns:
        True if toxic (label > 0), False if non-toxic (label == 0)
    """
    return label != ToxicityLabel.NON_TOXIC


def get_binary_labels(df: pd.DataFrame) -> pd.Series:
    """Convert multi-class toxicity labels to binary (toxic/non-toxic).

    Args:
        df: DataFrame with 'label' column

    Returns:
        Series of boolean values (True = toxic, False = non-toxic)
    """
    return df["label"].apply(is_toxic)


def load_reddit_usernames(
    data_path: str = "data/users.csv",
    sample_size: int | None = 10000,
    random_state: int = 42,
) -> pd.Series:
    """Load Reddit usernames dataset.

    Args:
        data_path: Path to users.csv file
        sample_size: Number of usernames to load. If None, load all (~25M usernames).
                    Default is 10,000 for faster processing.
        random_state: Random seed for sampling (for reproducibility)

    Returns:
        Series of Reddit usernames (author column only)

    Raises:
        FileNotFoundError: If the data file doesn't exist
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Reddit usernames dataset not found at {data_path}. "
            "Run 'cd data && bash download.sh' to download it."
        )

    df = pd.read_csv(path, usecols=["author"], nrows=sample_size)

    return df["author"]
