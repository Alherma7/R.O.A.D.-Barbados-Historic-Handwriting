"""End-to-end: load -> preprocess -> fit -> evaluate -> save predictions.

Skeleton only. Wire in src/features.py and src/model.py once an approach
has been validated in a notebook (see notebooks/01_eda_*).
"""

from src.data import load_raw_data, split_train_val


def main() -> None:
    train_df = load_raw_data("train")
    test_df = load_raw_data("test")  # noqa: F841 - wired in once a model exists

    fit_df, val_df = split_train_val(train_df)

    raise NotImplementedError(
        "Pick an OCR approach in src/model.py after comparing VLM/PaddleOCR/Kraken "
        "in a notebook, then fit on fit_df and predict on val_df/test_df here."
    )


if __name__ == "__main__":
    main()
