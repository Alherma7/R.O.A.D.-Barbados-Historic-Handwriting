# Resources

## Papers

- **Shi, B., Bai, X., & Yao, C. (2015). An End-to-End Trainable Neural Network for
  Image-based Sequence Recognition and Its Application to Scene Text Recognition.**
  (https://arxiv.org/abs/1507.05166)
  Why: source architecture for the from-scratch CRNN+CTC baseline in
  `notebooks/05_baseline_crnn.ipynb` / `src/model.py` — CNN feature extractor + BiLSTM +
  CTC loss/decoding, the standard approach for variable-length line-level OCR without a
  pretrained backbone. Requested as the first baseline before comparing the starter pack's
  pretrained approaches (VLM/PaddleOCR/Kraken).

## Comparable projects / official material

- **Zindi Barbados Road Challenge — starter pack** (`reference/starter_pack/`, provided by the competition)
  Why: source of the three baseline OCR approaches (VLM, PaddleOCR, Kraken). Its
  `eval_metrics.py` implied a `0.7*CER + 0.3*WER` leaderboard blend that turned out to be
  wrong — see the discussion thread below.
- **Zindi official discussion — R.O.A.D. Barbados scoring formula**
  (https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/33847)
  Why: competition staff (J0NNY) posted the actual public-leaderboard scoring function
  (`score = 1 - 0.5 * (wer_weighted/12 + cer_weighted/55)`, raw per-line Levenshtein
  distances, not length-normalized CER/WER) — the authoritative source for
  `src/evaluate.py`, replacing the incorrect `0.7*CER + 0.3*WER` approximation.

## Libraries

- **rapidfuzz** (https://github.com/rapidfuzz/RapidFuzz)
  Why: `distance.Levenshtein` computes the raw per-line word/character edit distances in
  `src/evaluate.py`, matching the exact function J0NNY posted in the Zindi discussion above.
  `fuzz.ratio` separately powers the near-duplicate/label-noise scan in
  `notebooks/04_eda_label_noise.ipynb` — infeasible at this line count with pure-Python
  `difflib`.
- **scikit-learn `GroupShuffleSplit`**
  (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html)
  Why: `src/data.py::split_train_val` groups on exact `Target` text (the only entity-like
  signal available, since Train.csv has no document/page column) so duplicate boilerplate
  lines can't leak across fit/val — validated in `notebooks/03_train_val_split.ipynb`.

## Models

- **Qwen2-VL-2B-Instruct** (https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
  Why: base model for the VLM approach in `reference/starter_pack/VLM/` (LoRA fine-tuned).
- **PaddleOCR** (https://github.com/PaddlePaddle/PaddleOCR)
  Why: detection + angle classification + recognition pipeline in `reference/starter_pack/Paddle-OCR/`.
- **Kraken** (https://kraken.re/)
  Why: OCR engine specialized for historical documents, used in `reference/starter_pack/Kraken-OCR/`.
