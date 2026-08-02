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
- 2026-08-02: Mitigated the baseline CRNN's overfitting — same
  `notebooks/05_baseline_crnn.ipynb`, three changes together (see
  RESOURCES.md): spatial/LSTM/pre-classifier dropout (Srivastava et al.
  2014, `DROPOUT=0.2`), `AdamW` with decoupled weight decay (Loshchilov &
  Hutter 2019, `WEIGHT_DECAY=1e-4`), and early stopping on the val score
  with best-checkpoint restore (Prechelt 1998, `MAX_EPOCHS=50`,
  `PATIENCE=8`). Trained and evaluated the same way as the baseline (same
  fit split, same fixed 120-row val sample), compared directly against
  the recorded 0.568 rather than a same-run from-scratch baseline (user's
  explicit choice — faster, at the cost of not isolating the
  eval-cadence/checkpoint-selection change from the regularization
  itself). **Result: `score = 0.691` at epoch 50 (`wer_weighted` 5.13,
  `cer_weighted` 10.42), +0.123 over the baseline.** Early stopping never
  triggered — the val score was still improving at `MAX_EPOCHS=50`, so
  this may not be the ceiling for this approach. Clears the gate
  (beats 0.568); saved as the new reference model,
  `models/crnn_regularized.pt`.
- 2026-08-02: Continued training `crnn_regularized.pt` for another
  up-to-50-epoch budget (loaded the checkpoint, fresh `AdamW` state, same
  `train_with_early_stopping` helper, `CONTINUE_PATIENCE=8`) instead of
  retraining from scratch with a higher `MAX_EPOCHS` — no LR scheduler is
  used, so a warm start from the checkpoint reaches the same place as a
  longer single run without repeating the first 50 epochs (only the
  optimizer's momentum state resets, not the weights). **Result: `score =
  0.702` (`wer_weighted` 4.98, `cer_weighted` 9.95), +0.011 over 0.691,
  best epoch 18/50.** Early stopping *did* trigger action this time (best
  found well before the epoch budget ran out) — much smaller gain than the
  first regularization pass, suggesting this architecture/setup is
  approaching its ceiling. Clears the gate; `models/crnn_regularized.pt`
  updated in place.

- 2026-08-02: Informal zero-shot sanity checks (8 val-sample rows, not the
  full 120-row comparison) for two starter-pack/off-the-shelf approaches,
  before deciding whether either is worth a full run or fine-tuning:
  - **PaddleOCR** (`PP-OCRv6_medium_rec`, recognition-only via
    `paddleocr.TextRecognition` — no page detection needed since our
    images are already single-line crops, unlike the starter pack's
    `inference.py` which assumes full-page input): `score = 0.393`
    (`wer_weighted` 8.0, `cer_weighted` 30.1). Failures look like garbled
    character-level noise (e.g. `'roloafodgint rlaynw and for ovor
    roufirnds anw by thod urofouts by'`), some empty predictions.
  - **TrOCR** (`microsoft/trocr-base-handwritten`, see RESOURCES.md):
    `score = 0.382` (`wer_weighted` 8.25, `cer_weighted` 30.1) — nearly
    tied with PaddleOCR numerically, but fails differently: fluent,
    grammatical modern-English sentences that are semantically invented
    (e.g. `"me was the Premier of Utah's calls called me then the basis
    of the subsequent"`), not character noise. Consistent with its
    RoBERTa-initialized decoder falling back on its modern-English
    language-model prior when faced with unfamiliar 17th-century cursive,
    rather than transcribing literally.
  - **Both far below the CRNN's 0.702**, and roughly tied with each
    other. Neither is competitive zero-shot; a real comparison would
    need fine-tuning (LoRA/full) rather than off-the-shelf weights,
    which is a much bigger investment than a sanity check.
  - Environment note: installing `paddleocr`+`paddlepaddle` and `kraken`
    into the shared Anaconda `base` environment (not an isolated
    per-approach env, unlike what the starter pack's own `setup.sh`
    scripts do) caused cascading dependency conflicts — `kraken`
    silently downgraded `torch` to a CPU-only build and bumped `numpy`
    to a version incompatible with other unrelated packages in the same
    environment (`gensim`, `numba`, `contourpy`, `streamlit`). Rolled
    back: uninstalled `paddleocr`/`paddlepaddle`/`paddlex`/`kraken` and
    their exclusive dependencies, restored `torch`/`torchvision` to a
    matching CUDA 12.6 build, and pinned `numpy`/`scipy`/`pillow`/
    `fsspec`/`safetensors`/`click` back to ranges compatible with the
    rest of the environment. No exact pre-session `pip freeze` was taken,
    so this is a best-effort, conflict-free restoration, not a
    guaranteed byte-for-byte one. Kraken was not tested (rolled back
    before a working check ran). Lesson for future approaches: install
    into an isolated environment per approach from the start.

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
- [x] Mitigate the CRNN baseline's overfitting (dropout + AdamW weight
      decay + early stopping) — `score = 0.691` (+0.123), new reference
      model `models/crnn_regularized.pt`, see Progress
- [x] Continue training the regularized CRNN from checkpoint instead of a
      higher-`MAX_EPOCHS` full retrain — `score = 0.702` (+0.011 over
      0.691), gains diminishing (best epoch 18/50 this time vs. never
      triggering early stopping the first pass) — likely near this
      approach's ceiling, see Progress
- [x] Zero-shot sanity check PaddleOCR and TrOCR (8-row informal sample,
      not the full 120) against the CRNN's 0.702 — both far behind
      (0.393 / 0.382), neither competitive off-the-shelf, see Progress
- [x] Decide next approach given the informal check — **user's call:
      fine-tune TrOCR only** (`notebooks/06_trocr_finetune.ipynb`);
      PaddleOCR and Kraken dropped (PaddleOCR's zero-shot gap was as bad
      as TrOCR's with more setup friction; Kraken untested after the
      environment rollback, not worth revisiting); VLM/Qwen2-VL LoRA
      dropped too (bigger investment, no zero-shot signal collected to
      justify it over TrOCR)
- [ ] Fine-tune TrOCR (`microsoft/trocr-base-handwritten`) on the fit
      split, compare on the same 120-row val sample against the CRNN's
      0.702 — in progress, `notebooks/06_trocr_finetune.ipynb`
- [ ] Consider data augmentation on the CRNN (rotation/elastic distortion/
      contrast) as a fallback if TrOCR fine-tuning doesn't beat 0.702
- [ ] Pick a winner (CRNN vs. fine-tuned TrOCR), wire it into
      `src/model.py`, graduate `src/train.py` from skeleton to a working
      end-to-end script
