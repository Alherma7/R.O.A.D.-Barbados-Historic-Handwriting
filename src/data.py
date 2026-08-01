"""Loading raw competition data and writing submission-shaped output."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    ID_COLUMN,
    IMAGE_EXTENSION,
    IMAGES_DIR,
    OUTPUTS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_CSV,
    TRAIN_CSV,
)


def load_raw_data(split: str) -> pd.DataFrame:
    """Load Train.csv or Test.csv as-is (ID [, Target])."""
    if split == "train":
        return pd.read_csv(TRAIN_CSV)
    if split == "test":
        return pd.read_csv(TEST_CSV)
    raise ValueError(f"Unknown split: {split!r}, expected 'train' or 'test'")


def build_image_path(record_id: str, images_dir: Path = IMAGES_DIR) -> Path:
    """Map a record ID to its expected image file, per the starter pack convention."""
    return images_dir / f"{record_id}{IMAGE_EXTENSION}"


def split_train_val(
    df: pd.DataFrame, val_size: float = 0.2, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split train rows into fit/val, keeping exact-duplicate Target text together.

    Train.csv has no document/page column to group on, so the closest
    available entity signal is exact-duplicate Target text (repeated legal
    boilerplate phrases). A plain random split leaks 3/10 of those
    duplicate groups across fit/val; grouping on Target text via
    GroupShuffleSplit (https://scikit-learn.org/stable/modules/generated/
    sklearn.model_selection.GroupShuffleSplit.html) eliminates that leakage
    with no measurable distortion to the length distribution — validated in
    notebooks/03_train_val_split.ipynb.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    fit_idx, val_idx = next(splitter.split(df, groups=df[TARGET_COLUMN]))
    return df.iloc[fit_idx], df.iloc[val_idx]


def save_output(df: pd.DataFrame, filename: str) -> Path:
    """Write a DataFrame with ID/Target columns to outputs/<filename>."""
    if list(df.columns) != [ID_COLUMN, TARGET_COLUMN]:
        raise ValueError(f"Expected columns {[ID_COLUMN, TARGET_COLUMN]}, got {list(df.columns)}")
    output_path = OUTPUTS_DIR / filename
    df.to_csv(output_path, index=False)
    return output_path
