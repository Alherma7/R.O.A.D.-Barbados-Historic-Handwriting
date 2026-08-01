"""This project's evaluation metric: the official Zindi public-leaderboard score.

score = 1 - SCORE_BLEND_WEIGHT * (wer_weighted / WER_NORM + cer_weighted / CER_NORM)

wer_weighted/cer_weighted are the *raw* (unnormalized) mean Levenshtein distance
per line - not CER/WER as error rates divided by reference length. Formula and
reference implementation confirmed by competition staff (J0NNY) on the official
Zindi discussion forum:
https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/33847
(see RESOURCES.md). This replaces an earlier, incorrect approximation based on
0.7*CER + 0.3*WER via jiwer's corpus-level, length-normalized error rates, copied
from reference/starter_pack/*/eval_metrics.py before the real formula was confirmed.

Note the direction: this is a score (higher is better), the opposite of CER/WER
themselves (lower is better) - wer_weighted/cer_weighted are still returned here
since they're useful error diagnostics on their own.
"""

import pandas as pd
from rapidfuzz.distance import Levenshtein

from src.config import CER_NORM, ID_COLUMN, SCORE_BLEND_WEIGHT, TARGET_COLUMN, WER_NORM


def compute_wer_weighted(references: list[str], hypotheses: list[str]) -> float:
    """Mean word-level Levenshtein distance per line (raw edit count, not a rate)."""
    return sum(
        Levenshtein.distance(ref.split(), hyp.split()) for ref, hyp in zip(references, hypotheses)
    ) / len(references)


def compute_cer_weighted(references: list[str], hypotheses: list[str]) -> float:
    """Mean character-level Levenshtein distance per line (raw edit count, not a rate)."""
    return sum(
        Levenshtein.distance(ref, hyp) for ref, hyp in zip(references, hypotheses)
    ) / len(references)


def combined_score(references: list[str], hypotheses: list[str]) -> dict:
    """wer_weighted, cer_weighted and the official public-leaderboard score (higher is better)."""
    wer_weighted = compute_wer_weighted(references, hypotheses)
    cer_weighted = compute_cer_weighted(references, hypotheses)
    score = 1 - SCORE_BLEND_WEIGHT * (wer_weighted / WER_NORM + cer_weighted / CER_NORM)
    return {
        "wer_weighted": wer_weighted,
        "cer_weighted": cer_weighted,
        "score": score,
    }


def score_submission(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Score a predictions DataFrame (ID, Target) against ground truth (ID, Target)."""
    merged = ground_truth.merge(predictions, on=ID_COLUMN, suffixes=("_true", "_pred"))
    references = merged[f"{TARGET_COLUMN}_true"].fillna("").tolist()
    hypotheses = merged[f"{TARGET_COLUMN}_pred"].fillna("").tolist()
    return combined_score(references, hypotheses)
