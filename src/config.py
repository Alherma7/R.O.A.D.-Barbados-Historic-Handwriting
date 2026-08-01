"""Paths, constants and metric weights shared across the pipeline."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
IMAGES_DIR = DATA_RAW_DIR / "images"

MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

TRAIN_CSV = DATA_RAW_DIR / "Train.csv"
TEST_CSV = DATA_RAW_DIR / "Test.csv"
SAMPLE_SUBMISSION_CSV = DATA_RAW_DIR / "SampleSubmission.csv"

ID_COLUMN = "ID"
TARGET_COLUMN = "Target"
IMAGE_EXTENSION = ".jpg"

RANDOM_STATE = 42

# Official public-leaderboard formula, confirmed by competition staff (J0NNY) on the
# Zindi discussion forum (see RESOURCES.md):
#   score = 1 - SCORE_BLEND_WEIGHT * (wer_weighted / WER_NORM + cer_weighted / CER_NORM)
# where wer_weighted/cer_weighted are the *raw* (unnormalized) mean Levenshtein distance
# per line, not CER/WER as error rates divided by reference length. Replaces an earlier,
# incorrect 0.7*CER + 0.3*WER approximation copied from reference/starter_pack/*/eval_metrics.py
# before the real formula was confirmed.
WER_NORM = 12
CER_NORM = 55
SCORE_BLEND_WEIGHT = 0.5
