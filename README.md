# Barbados Road Challenge (Zindi)

OCR competition: transcribe handwritten lines from historical Barbados
documents. 4,098 train lines / 1,374 test lines, `ID,Target` format.

Leaderboard metric: `score = 1 - 0.5 * (wer_weighted/12 + cer_weighted/55)`,
where `wer_weighted`/`cer_weighted` are the *raw* (unnormalized) mean
Levenshtein distance per line — confirmed by competition staff on the
[official Zindi discussion](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/33847)
(higher score is better). Implemented in `src/evaluate.py` via
`rapidfuzz.distance.Levenshtein`. This replaced an earlier, incorrect
`0.7*CER + 0.3*WER` approximation copied from
`reference/starter_pack/*/eval_metrics.py` — see Progress.

Three candidate approaches ship in `reference/starter_pack/`: VLM
(Qwen2-VL, LoRA fine-tune), PaddleOCR, Kraken-OCR. None chosen yet —
compare them in a notebook against `src/evaluate.py::combined_score`
before wiring one into `src/model.py`.

## Progress

- 2026-08-01: Project scaffolded (skeleton, config, data loading,
  evaluate.py with CER/WER/combined_score + tests). No model work yet.
- 2026-08-01: Competition image archive downloaded into `data/raw/images/`
  — 5,472 files, exact 1:1 coverage with Train+Test IDs (0 missing, 0
  extra).
- 2026-08-01: EDA done — `notebooks/01_eda_text_overview.ipynb` (text) and
  `notebooks/02_eda_images.ipynb` (images), both executed with saved
  outputs. Key findings:
  - Target length: 37–120 chars, mean 62. Character set is plain ASCII
    letters/digits/punctuation plus a handful of transcription markup
    chars (`# * ^ _ \ | ~`) — likely convention for illegible/uncertain
    text; worth checking the starter pack docs before treating these as
    noise.
  - Exact-duplicate `Target` lines: only 10 of 4,086 unique lines repeat
    (max count 3, all common legal boilerplate phrases e.g. "Sealed and
    delivered in the presence of"). Low leakage risk from duplicate text.
  - **`Train.csv`/`Test.csv` have no document/page column** — only `ID`,
    `Target`. There's no direct signal to group rows by source document,
    so a group-aware split isn't implementable as originally planned; see
    Next steps.
  - Images: all RGB JPEG, no corrupt/unreadable files. Width and height
    vary far more than expected for tight single-line crops (width
    267–6051px, height 28–1131px, aspect ratio 3–40) — and **width/height
    are *negatively* correlated with transcription length**
    (corr ≈ -0.12 / -0.22). This means pixel size is not a simple proxy
    for text length, most likely because crops come from source documents
    scanned at different resolutions/DPI. Any width-based
    preprocessing (e.g. resize-to-aspect-ratio) needs to account for this
    per-document scale variation rather than assuming a fixed
    px-per-character ratio.
- 2026-08-01: Manually reviewed the width/height outlier images in
  `notebooks/02_eda_images.ipynb` (the extremes from the describe() above)
  and found the size variation isn't just "more/less text":
  - A handful of the smallest-width crops are cramped, stacked multi-line
    signature/witness-mark blocks (e.g. `PO7QQLWIFOT65BTz`: "the marks of
    Thomas I Mosier..." across 2-3 short lines) — the single-line-crop
    assumption doesn't hold for every row.
  - Several outliers show bleed-through from the adjacent line at the crop
    edge (extra ink not reflected in the `Target` transcription).
  - One height outlier (`gSRiLOj8cDjjmcYn`) looks like a land-survey/plat
    annotation ("Land Staked out for Mr. John Gardner...") rather than
    deed prose — a distinct sub-population in the corpus.
  - Not a data error to fix, but worth remembering during model error
    analysis so these known-hard cases (multi-line crops, bleed-through,
    non-prose content) aren't mistaken for generic model failures.
- 2026-08-01: Train/val split strategy decided in
  `notebooks/03_train_val_split.ipynb` and wired into `src/data.py`. With
  no document/page column available, a plain random split leaks 3/10
  exact-duplicate-text groups across fit/val; a `GroupShuffleSplit`
  grouped on exact `Target` text leaks 0 with no measurable distortion to
  the length distribution. Adopted as `src/data.py::split_train_val`
  (tested in `tests/test_data.py`), replacing the plain `train_test_split`
  call in `src/train.py`.
- 2026-08-01: Investigated a manually-spotted typo (`Sealed and delivered
  in the prsence of` vs `...presence of`) with a systematic near-duplicate
  scan (`notebooks/04_eda_label_noise.ipynb`, rapidfuzz `fuzz.ratio` over
  all train line pairs). Findings:
  - Most near-duplicate lines (423 fuzzy pairs / 50 clusters at ratio
    >=88) are genuine repeated legal boilerplate across different source
    documents with real period spelling/abbreviation variation (`Xpian`,
    `pnts`, `p^rsent`) — not errors, and not something to normalize.
  - `presence`/`prsence` looked different at first: all 19 `prsence`-
    labeled source images show the same compressed cursive ligature as
    the correctly-transcribed `presence` rows, with no abbreviation mark,
    in the same standard "sealed and delivered in the presence of
    witnesses" formula. Initially read this as a confirmed transcriber
    typo and applied a targeted correction
    (`src/data.py::TARGET_CORRECTIONS`, 19 explicit ID -> corrected-text
    entries) — **but reverted it** after a second look at the source image
    used to justify it (`zF6Y0Q8CW0z7x0EU`) showed the "unambiguously
    reads presence" claim didn't hold up; the ligature is genuinely hard
    to disambiguate either way.
  - **Decision: do not normalize/correct `Target` spelling anywhere in the
    pipeline**, including this specific case — same reasoning as the
    broader boilerplate clusters, now on firmer footing since we can't
    actually confirm which spelling (if either) is "correct" from the
    image alone, and `src/evaluate.py` scores against whatever reference
    text Zindi provides. Full near-duplicate cluster table still saved to
    `data/processed/near_duplicate_clusters.csv` for reference during
    later model error analysis.
- 2026-08-01: **Corrected the evaluation metric** — found and confirmed the
  actual Zindi public-leaderboard formula via the
  [official discussion forum](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/33847)
  (competition staff posted the exact scoring function). It is *not*
  `0.7*CER + 0.3*WER` on length-normalized error rates (what
  `reference/starter_pack/*/eval_metrics.py` implied and what
  `src/evaluate.py` implemented until now). The real formula:
  `score = 1 - 0.5 * (wer_weighted/12 + cer_weighted/55)`, where
  `wer_weighted`/`cer_weighted` are the *raw* (unnormalized) mean
  Levenshtein distance per line via `rapidfuzz.distance.Levenshtein`, and
  higher `score` is better. Rewrote `src/evaluate.py` and
  `tests/test_evaluate.py` (new key names: `wer_weighted`, `cer_weighted`,
  `score`, replacing `cer`/`wer`/`final`) to match. Every score computed
  before this point (e.g. any CRNN baseline numbers from
  `notebooks/05_baseline_crnn.ipynb` run before this fix) used the wrong
  metric and needs to be recomputed.
- 2026-08-01: First baseline — CRNN+CTC trained from scratch (no
  pretrained weights), per user request, before comparing the starter
  pack's pretrained approaches. `notebooks/05_baseline_crnn.ipynb`:
  CNN + 2-layer BiLSTM + CTC (Shi, Bai & Yao 2015, see RESOURCES.md),
  trained ~30+ epochs on the full fit split (3,280 rows), evaluated with
  the corrected `src/evaluate.py` on the fixed 120-row val sample.
  **Result: `score = 0.568` (`wer_weighted` 7.01, `cer_weighted` 15.39).**
  Train loss kept dropping while val score plateaued/CER ticked up
  slightly around epoch 10-15 — mild overfitting, expected training every
  weight from scratch on only 3,280 examples. Left as-is (no dropout/
  weight-decay/early-stopping added) to keep this a clean reference
  number; revisit only if we later want to push this specific approach
  further. Model weights saved to `models/crnn_baseline.pt`. This is the
  number the starter-pack approaches (VLM/PaddleOCR/Kraken) need to beat
  on the same 120-row val sample to be worth adopting.

## Next steps

- [x] Download the competition's image archive from Zindi's Data page into
      `data/raw/images/` (`<ID>.jpg`, see `data/raw/images/.gitkeep`)
- [x] `notebooks/01_eda_*`: image sizes/aspect ratios, character set
      coverage, line-length distribution, any duplicate/near-duplicate
      lines across train
- [x] Decide on a train/validation split strategy — `GroupShuffleSplit` on
      exact `Target` text, see Progress and
      `notebooks/03_train_val_split.ipynb`
- [x] From-scratch CRNN+CTC baseline — `score = 0.568` on the 120-row val
      sample, see Progress and `notebooks/05_baseline_crnn.ipynb`
- [x] Locate/confirm Zindi's actual scoring formula — done via the
      [official discussion forum](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/33847),
      see Progress ("Corrected the evaluation metric")
- [ ] Run PaddleOCR baseline (zero-shot, no training needed) on the same
      120-row val sample, compare `score` against the CRNN's 0.568
- [ ] Run Kraken baseline (pretrained model download + inference) on the
      same val sample
- [ ] Run VLM (Qwen2-VL LoRA) baseline — needs a short fine-tune first
      since `reference/starter_pack/VLM/inference.py` requires a trained
      adapter
- [ ] Pick a winner among CRNN/PaddleOCR/Kraken/VLM, wire it into
      `src/model.py`, graduate `src/train.py` from skeleton to a working
      end-to-end script
