import pandas as pd
import pytest

from src.config import CER_NORM, SCORE_BLEND_WEIGHT, WER_NORM
from src.evaluate import combined_score, compute_cer_weighted, compute_wer_weighted, score_submission


def test_compute_wer_weighted_identical_lines_is_zero():
    assert compute_wer_weighted(["hello world"], ["hello world"]) == pytest.approx(0.0)


def test_compute_cer_weighted_identical_lines_is_zero():
    assert compute_cer_weighted(["hello world"], ["hello world"]) == pytest.approx(0.0)


def test_compute_cer_weighted_is_raw_edit_distance_not_a_rate():
    # "cat" -> "car": 1 raw character substitution, not divided by reference length.
    assert compute_cer_weighted(["cat"], ["car"]) == pytest.approx(1.0)


def test_compute_wer_weighted_is_raw_edit_distance_not_a_rate():
    # "the cat sat" -> "the car sat": 1 raw word substitution, not divided by word count.
    assert compute_wer_weighted(["the cat sat"], ["the car sat"]) == pytest.approx(1.0)


def test_compute_wer_weighted_averages_across_lines():
    # line 1: 0 edits, line 2: 1 word substitution -> mean = 0.5
    references = ["the cat sat", "a dog ran"]
    hypotheses = ["the cat sat", "a dog sat"]
    assert compute_wer_weighted(references, hypotheses) == pytest.approx(0.5)


def test_combined_score_blends_wer_and_cer_per_official_formula():
    # "cat" -> "car": word-level distance 1 (whole word differs), char-level distance 1.
    result = combined_score(["cat"], ["car"])
    expected_wer_weighted = 1.0
    expected_cer_weighted = 1.0
    expected_score = 1 - SCORE_BLEND_WEIGHT * (expected_wer_weighted / WER_NORM + expected_cer_weighted / CER_NORM)
    assert result["wer_weighted"] == pytest.approx(expected_wer_weighted)
    assert result["cer_weighted"] == pytest.approx(expected_cer_weighted)
    assert result["score"] == pytest.approx(expected_score)


def test_combined_score_identical_lines_scores_maximum():
    result = combined_score(["hello world"], ["hello world"])
    assert result["score"] == pytest.approx(1.0)


def test_score_submission_merges_on_id_and_handles_missing_prediction():
    ground_truth = pd.DataFrame({"ID": ["a", "b"], "Target": ["cat", "dog"]})
    predictions = pd.DataFrame({"ID": ["a", "b"], "Target": ["cat", ""]})

    result = score_submission(ground_truth, predictions)

    assert result["cer_weighted"] > 0.0
    assert result["score"] < 1.0
