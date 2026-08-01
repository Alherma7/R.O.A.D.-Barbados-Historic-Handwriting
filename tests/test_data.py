from pathlib import Path

import pandas as pd
import pytest

from src.data import build_image_path, save_output, split_train_val
from src.config import ID_COLUMN, TARGET_COLUMN


def test_build_image_path_appends_jpg_extension(tmp_path):
    result = build_image_path("abc123", images_dir=tmp_path)
    assert result == tmp_path / "abc123.jpg"


def test_save_output_writes_id_target_csv(tmp_path, monkeypatch):
    import src.data as data_module

    monkeypatch.setattr(data_module, "OUTPUTS_DIR", tmp_path)
    df = pd.DataFrame({ID_COLUMN: ["a", "b"], TARGET_COLUMN: ["cat", "dog"]})

    output_path = save_output(df, "submission.csv")

    assert output_path == tmp_path / "submission.csv"
    written = pd.read_csv(output_path)
    pd.testing.assert_frame_equal(written, df)


def test_save_output_rejects_wrong_columns(tmp_path, monkeypatch):
    import src.data as data_module

    monkeypatch.setattr(data_module, "OUTPUTS_DIR", tmp_path)
    df = pd.DataFrame({"id": ["a"], "prediction": ["cat"]})

    with pytest.raises(ValueError):
        save_output(df, "submission.csv")


def _make_split_fixture() -> pd.DataFrame:
    # 20 rows: one duplicate Target repeated 3x, one repeated 2x, rest unique -
    # mirrors the boilerplate-phrase duplicates found in the real Train.csv.
    targets = (
        ["Sealed and delivered in the presence of"] * 3
        + ["By this publique Act and Instrument"] * 2
        + [f"unique line {i}" for i in range(15)]
    )
    return pd.DataFrame({ID_COLUMN: [f"id{i}" for i in range(len(targets))], TARGET_COLUMN: targets})


def test_split_train_val_keeps_duplicate_targets_together():
    df = _make_split_fixture()

    fit_df, val_df = split_train_val(df, val_size=0.2, random_state=0)

    for target, group in df.groupby(TARGET_COLUMN):
        in_fit = group[ID_COLUMN].isin(fit_df[ID_COLUMN]).any()
        in_val = group[ID_COLUMN].isin(val_df[ID_COLUMN]).any()
        assert not (in_fit and in_val), f"{target!r} split across fit and val"


def test_split_train_val_preserves_all_rows_with_no_overlap():
    df = _make_split_fixture()

    fit_df, val_df = split_train_val(df, val_size=0.2, random_state=0)

    assert set(fit_df[ID_COLUMN]).isdisjoint(set(val_df[ID_COLUMN]))
    assert len(fit_df) + len(val_df) == len(df)
    assert set(fit_df[ID_COLUMN]) | set(val_df[ID_COLUMN]) == set(df[ID_COLUMN])
