import torch

from src.features import ctc_output_length
from src.model import CharNgramLM, CRNNResidual, beam_search_decode, greedy_decode

IDX_TO_CHAR = {1: "a", 2: "b"}
BLANK = 0


def test_crnn_residual_forward_pass_shape():
    model = CRNNResidual(num_classes=5, dropout=0.0)
    model.eval()
    x = torch.randn(2, 1, 32, 64)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, ctc_output_length(64), 5)


def test_crnn_residual_output_is_log_probabilities():
    model = CRNNResidual(num_classes=5, dropout=0.0)
    model.eval()
    x = torch.randn(1, 1, 32, 64)

    with torch.no_grad():
        out = model(x)

    # log_softmax output: each timestep's probabilities sum to 1.
    assert torch.allclose(out.exp().sum(dim=-1), torch.ones(1, out.shape[1]), atol=1e-4)


def _one_hot_log_probs(path, num_classes=3, hot=20.0, cold=0.0):
    logits = torch.full((len(path), num_classes), cold)
    for t, c in enumerate(path):
        logits[t, c] = hot
    return logits.log_softmax(-1)


def test_greedy_decode_collapses_repeats_and_drops_blanks():
    # a, a, blank, a -> collapse the leading repeat, drop the blank, keep the
    # trailing 'a' (a new symbol after a blank always counts again).
    log_probs = _one_hot_log_probs([1, 1, BLANK, 1]).unsqueeze(0)
    out = greedy_decode(log_probs, input_lengths=[4], idx_to_char=IDX_TO_CHAR, blank=BLANK)
    assert out == ["aa"]


def test_beam_search_decode_matches_known_ctc_collapse():
    # Same deterministic case validated in notebooks/08_decoding_strategies.ipynb:
    # near-one-hot logits force the argmax path a, a, blank, a -> "aa".
    log_probs = _one_hot_log_probs([1, 1, BLANK, 1])
    out = beam_search_decode(log_probs, input_length=4, idx_to_char=IDX_TO_CHAR, blank=BLANK, beam_width=5, top_k=3)
    assert out == "aa"


def test_beam_search_decode_lm_fusion_changes_the_decision():
    # Ambiguous per-frame probabilities (no single dominant path) - an LM
    # strongly favoring "ab"-repeating text should be able to flip the
    # decoded output relative to no LM at all (alpha=0).
    lm = CharNgramLM(["aabaabaabaab"] * 20 + ["abbabbabbabb"], n=3, k=1e-3)
    log_probs = torch.log(
        torch.tensor(
            [
                [0.05, 0.50, 0.45],
                [0.05, 0.45, 0.50],
                [0.05, 0.48, 0.47],
            ]
        )
    )

    out_no_lm = beam_search_decode(log_probs, 3, IDX_TO_CHAR, blank=BLANK, beam_width=10, top_k=3, lm=None, alpha=0.0)
    out_with_lm = beam_search_decode(
        log_probs, 3, IDX_TO_CHAR, blank=BLANK, beam_width=10, top_k=3, lm=lm, alpha=2.0
    )

    assert out_no_lm != out_with_lm


def test_beam_search_decode_alpha_zero_ignores_the_lm():
    lm = CharNgramLM(["aabaabaabaab"] * 20 + ["abbabbabbabb"], n=3, k=1e-3)
    log_probs = _one_hot_log_probs([1, 1, BLANK, 1])

    with_zero_alpha = beam_search_decode(
        log_probs, 4, IDX_TO_CHAR, blank=BLANK, beam_width=5, top_k=3, lm=lm, alpha=0.0
    )
    without_lm = beam_search_decode(log_probs, 4, IDX_TO_CHAR, blank=BLANK, beam_width=5, top_k=3, lm=None, alpha=0.0)

    assert with_zero_alpha == without_lm


def test_char_ngram_lm_prefers_seen_continuations():
    lm = CharNgramLM(["aab"] * 10, n=3, k=1e-3)
    # After context "aa", 'b' was seen every time - it should score higher
    # than a character that never followed "aa" in training.
    assert lm.get_log_prob("b", "aa") > lm.get_log_prob("z", "aa")
