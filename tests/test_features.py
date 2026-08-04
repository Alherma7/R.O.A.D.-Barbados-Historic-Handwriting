import torch

from src.features import build_vocab, collate_lines, ctc_output_length, elastic_distort


def test_build_vocab_reserves_zero_for_blank():
    char_to_idx, idx_to_char = build_vocab(["ba", "ab"])
    assert 0 not in idx_to_char
    assert set(char_to_idx.values()) == {1, 2}


def test_build_vocab_covers_every_character_and_is_deterministic():
    texts = ["hello world", "foo bar"]
    char_to_idx, idx_to_char = build_vocab(texts)
    assert set(char_to_idx) == set("".join(texts))
    assert char_to_idx == build_vocab(texts)[0]  # same input -> same mapping


def test_build_vocab_idx_to_char_is_the_inverse_of_char_to_idx():
    char_to_idx, idx_to_char = build_vocab(["xyz"])
    for char, idx in char_to_idx.items():
        assert idx_to_char[idx] == char


def test_ctc_output_length_matches_two_maxpool_2x2_stages():
    # (w // 2) // 2 - must match CRNNResidual's two MaxPool2d(2, 2) stages.
    assert ctc_output_length(64) == 16
    assert ctc_output_length(8) == 2


def test_ctc_output_length_rounds_down_on_odd_widths():
    # Verified against a real forward pass in notebooks/07: odd widths
    # round down at each pooling stage, not up.
    assert ctc_output_length(101) == 25


def test_elastic_distort_alpha_zero_reproduces_input():
    img = torch.randn(1, 16, 20)
    out = elastic_distort(img, alpha=0.0, sigma=4.0)
    assert torch.allclose(out, img, atol=1e-5)


def test_elastic_distort_preserves_shape():
    img = torch.randn(1, 32, 101)
    out = elastic_distort(img, alpha=12.0, sigma=4.0)
    assert out.shape == img.shape


def test_collate_lines_pads_to_batch_max_width_and_computes_input_lengths():
    img_a = torch.zeros(1, 32, 64)
    img_b = torch.zeros(1, 32, 100)
    batch = [
        (img_a, torch.tensor([1, 2], dtype=torch.long), "id_a", "ab"),
        (img_b, torch.tensor([3], dtype=torch.long), "id_b", "c"),
    ]

    padded, labels_cat, input_lengths, target_lengths, ids, texts = collate_lines(batch)

    assert padded.shape == (2, 1, 32, 100)
    assert input_lengths.tolist() == [ctc_output_length(64), ctc_output_length(100)]
    assert target_lengths.tolist() == [2, 1]
    assert labels_cat.tolist() == [1, 2, 3]
    assert ids == ["id_a", "id_b"]
    assert texts == ["ab", "c"]


def test_collate_lines_handles_missing_targets():
    img_a = torch.zeros(1, 32, 64)
    batch = [(img_a, torch.tensor([], dtype=torch.long), "id_a", None)]

    _padded, labels_cat, _input_lengths, target_lengths, ids, texts = collate_lines(batch)

    assert texts == [None]
    assert ids == ["id_a"]
    assert labels_cat.numel() == 0
    assert target_lengths.tolist() == [0]
