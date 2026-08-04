"""CRNN+CTC OCR model: architecture, beam-search+LM decoding, and inference.

Architecture: CNN feature extractor + BiLSTM + CTC head (Shi, Bai & Yao
2015, https://arxiv.org/abs/1507.05166), with a residual block per CNN
stage (He, Zhang, Ren & Sun 2015, https://arxiv.org/abs/1512.03385) - the
winning variant from notebooks/07_crnn_architecture_variants.ipynb (0.720
vs. 0.702 for the plain baseline, same val sample).

Decoding: CTC prefix beam search (Graves, Fernandez, Gomez & Schmidhuber
2006; algorithm per Hannun 2017, https://distill.pub/2017/ctc/) with
character n-gram language-model shallow fusion (Amodei et al. 2016, "Deep
Speech 2", https://arxiv.org/abs/1512.02595) - validated in
notebooks/08_decoding_strategies.ipynb (0.796 -> 0.826 at alpha=0.35 on
the residual+augmentation+elastic-distortion checkpoint). See RESOURCES.md
for every citation above.
"""

import math
from collections import Counter, defaultdict

import torch
import torch.nn as nn

NEG_INF = float("-inf")


class ResidualBlock(nn.Module):
    """Two 3x3 convs with a skip connection; a 1x1 conv projects the shortcut
    when the channel count changes (He et al. 2015 - see RESOURCES.md)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class CRNNResidual(nn.Module):
    """CNN (residual blocks) + BiLSTM + CTC head - the project's reference
    architecture (notebooks/07: 0.720 same-run vs. 0.702 plain-baseline
    reference). Dropout (Srivastava et al. 2014) and AdamW decoupled
    weight decay (Loshchilov & Hutter 2019, applied to the optimizer, not
    here) mitigate overfitting on the 3,280-row fit split - see
    RESOURCES.md."""

    def __init__(self, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            ResidualBlock(1, 32), nn.MaxPool2d(2, 2),
            ResidualBlock(32, 64), nn.MaxPool2d(2, 2),
            ResidualBlock(64, 128), nn.MaxPool2d((2, 1)),
            nn.Dropout2d(dropout),
            ResidualBlock(128, 128), nn.MaxPool2d((2, 1)),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, 256, (2, 1), 1, 0), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(256, 128, num_layers=2, bidirectional=True, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.squeeze(2).permute(0, 2, 1)
        seq, _ = self.rnn(feat)
        seq = self.dropout(seq)
        return self.fc(seq).log_softmax(2)


def greedy_decode(log_probs, input_lengths, idx_to_char: dict[int, str], blank: int = 0) -> list[str]:
    """Per-timestep argmax, collapse repeats, drop blanks (Graves et al. 2006)."""
    pred_idx = log_probs.argmax(2)
    texts = []
    for b in range(pred_idx.size(0)):
        seq = pred_idx[b, : input_lengths[b]].tolist()
        out, prev = [], None
        for idx in seq:
            if idx != prev and idx != blank:
                out.append(idx_to_char[idx])
            prev = idx
        texts.append("".join(out))
    return texts


class CharNgramLM:
    """Character n-gram LM, add-k (Laplace) smoothed counting, trained on
    this project's own fit-split Target text - not KenLM, avoids a C++
    native-dependency install for a corpus this small (~200K chars). Used
    for CTC beam search shallow fusion (Amodei et al. 2016 - see
    RESOURCES.md)."""

    def __init__(self, texts: list[str], n: int = 4, k: float = 1e-3):
        self.n = n
        self.k = k
        self.ngrams: dict[str, Counter] = defaultdict(Counter)
        self.context_counts: Counter = Counter()
        self.vocab: set[str] = set()
        for text in texts:
            padded = "^" * (n - 1) + text + "$"
            for i in range(len(padded) - n + 1):
                context = padded[i : i + n - 1]
                char = padded[i + n - 1]
                self.ngrams[context][char] += 1
                self.context_counts[context] += 1
                self.vocab.add(char)
        self.vocab_size = len(self.vocab)

    def get_log_prob(self, char: str, context: str) -> float:
        if len(context) > self.n - 1:
            context = context[-(self.n - 1) :]
        elif len(context) < self.n - 1:
            context = "^" * (self.n - 1 - len(context)) + context
        total = self.context_counts[context]
        count = self.ngrams[context][char]
        prob = (count + self.k) / (total + self.k * self.vocab_size)
        return math.log(prob)


def _logsumexp(*args: float) -> float:
    m = max(args)
    if m == NEG_INF:
        return NEG_INF
    return m + math.log(sum(math.exp(a - m) for a in args))


def beam_search_decode(
    log_probs,
    input_length: int,
    idx_to_char: dict[int, str],
    blank: int = 0,
    beam_width: int = 10,
    top_k: int = 25,
    lm: "CharNgramLM | None" = None,
    alpha: float = 0.0,
) -> str:
    """CTC prefix beam search, optionally with character-LM shallow fusion
    (Graves et al. 2006; Amodei et al. 2016 for the fusion formula;
    algorithm per Hannun 2017 - see RESOURCES.md). Tracks p_b/p_nb (blank-
    vs non-blank-ending paths) separately so repeated characters collapse
    per the real CTC rule, and merges duplicate prefixes via logsumexp
    (sums probability mass over every alignment, not max) - verified
    against a brute-force CTC forward score in notebooks/08. LM score is
    added once per newly emitted symbol:
    score = log P_ctc + alpha * log P_lm."""
    log_probs = log_probs[:input_length]
    T, C = log_probs.shape
    k = min(top_k, C)
    beams = {(): (0.0, NEG_INF)}

    for t in range(T):
        topk_logp, topk_idx = log_probs[t].topk(k)
        next_beams = {}

        for prefix, (p_b, p_nb) in beams.items():
            p_total = _logsumexp(p_b, p_nb)
            end_char = prefix[-1] if prefix else None
            prefix_str = "".join(idx_to_char[x] for x in prefix)

            for log_p, c in zip(topk_logp.tolist(), topk_idx.tolist()):
                if c == blank:
                    nb = next_beams.get(prefix, (NEG_INF, NEG_INF))
                    next_beams[prefix] = (_logsumexp(nb[0], p_total + log_p), nb[1])
                elif c == end_char:
                    nb = next_beams.get(prefix, (NEG_INF, NEG_INF))
                    next_beams[prefix] = (nb[0], _logsumexp(nb[1], p_nb + log_p))
                    lm_score = alpha * lm.get_log_prob(idx_to_char[c], prefix_str) if lm else 0.0
                    new_prefix = prefix + (c,)
                    nb2 = next_beams.get(new_prefix, (NEG_INF, NEG_INF))
                    next_beams[new_prefix] = (nb2[0], _logsumexp(nb2[1], p_b + log_p + lm_score))
                else:
                    lm_score = alpha * lm.get_log_prob(idx_to_char[c], prefix_str) if lm else 0.0
                    new_prefix = prefix + (c,)
                    nb2 = next_beams.get(new_prefix, (NEG_INF, NEG_INF))
                    next_beams[new_prefix] = (nb2[0], _logsumexp(nb2[1], p_total + log_p + lm_score))

        beams = dict(sorted(next_beams.items(), key=lambda kv: _logsumexp(*kv[1]), reverse=True)[:beam_width])

    best_prefix = max(beams.items(), key=lambda kv: _logsumexp(*kv[1]))[0]
    return "".join(idx_to_char[c] for c in best_prefix)


def beam_decode_batch(
    log_probs,
    input_lengths,
    idx_to_char: dict[int, str],
    blank: int = 0,
    beam_width: int = 10,
    lm: "CharNgramLM | None" = None,
    alpha: float = 0.0,
) -> list[str]:
    return [
        beam_search_decode(
            log_probs[b], input_lengths[b], idx_to_char, blank=blank, beam_width=beam_width, lm=lm, alpha=alpha
        )
        for b in range(log_probs.size(0))
    ]


def predict(
    model: nn.Module,
    dataloader,
    idx_to_char: dict[int, str],
    device,
    blank: int = 0,
    beam_width: int = 10,
    lm: "CharNgramLM | None" = None,
    alpha: float = 0.0,
) -> tuple[list[str], list[str], list[str | None]]:
    """Runs `model` over `dataloader`, decoding each batch with beam search
    (+ optional LM fusion). Returns (ids, hypotheses, references) aligned
    1:1 - `references` entries are None for rows with no Target (Test.csv).
    This is the project's default inference path, validated end-to-end in
    notebooks/08 (0.826 on the 120-row val sample, 0.82658842 on the Zindi
    leaderboard - see RESOURCES.md/README.md)."""
    model.eval()
    all_ids, all_hyps, all_refs = [], [], []
    with torch.no_grad():
        for imgs, _targets_cat, input_lengths, _target_lengths, ids, texts in dataloader:
            imgs = imgs.to(device)
            log_probs = model(imgs).cpu()
            hyps = beam_decode_batch(
                log_probs, input_lengths, idx_to_char, blank=blank, beam_width=beam_width, lm=lm, alpha=alpha
            )
            all_ids.extend(ids)
            all_hyps.extend(hyps)
            all_refs.extend(texts)
    return all_ids, all_hyps, all_refs
