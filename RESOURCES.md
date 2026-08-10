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
- **Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).
  Dropout: A Simple Way to Prevent Neural Networks from Overfitting.** JMLR.
  (https://jmlr.org/papers/v15/srivastava14a.html)
  Why: source for the regularization added to `CRNN` in
  `notebooks/05_baseline_crnn.ipynb` to mitigate the from-scratch baseline's overfitting
  (`nn.Dropout2d` after the last two CNN blocks, inter-layer LSTM dropout, dropout before
  the final classifier).
- **Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization.** ICLR.
  (https://arxiv.org/abs/1711.05101)
  Why: source for using `AdamW` (decoupled weight decay) instead of `Adam` + L2 in the same
  regularization pass, so weight decay doesn't get folded into the adaptive gradient scaling.
- **Prechelt, L. (1998). Early Stopping — But When?** In *Neural Networks: Tricks of the
  Trade*, Springer.
  (https://link.springer.com/chapter/10.1007/3-540-49430-8_3)
  Why: source for the early-stopping criterion (stop after `PATIENCE` epochs with no
  validation-score improvement, keep the best checkpoint) added alongside dropout/weight
  decay in the same notebook.

- **Li, M. et al. (2021). TrOCR: Transformer-based Optical Character Recognition with
  Pre-trained Models.** (https://arxiv.org/abs/2109.10282)
  Why: source for `microsoft/trocr-base-handwritten`, used as a zero-shot sanity check
  against the CRNN in `notebooks/05_baseline_crnn.ipynb`'s comparison — encoder-decoder
  architecture (ViT encoder + RoBERTa-initialized decoder), pretrained on modern English
  handwriting (IAM dataset).
- **Springenberg, J. T., Dosovitskiy, A., Brox, T., & Riedmiller, M. (2015). Striving for
  Simplicity: The All Convolutional Net.** (https://arxiv.org/abs/1412.6806)
  Why: source for `CRNNStride2` in `notebooks/07_crnn_architecture_variants.ipynb` —
  replaces every `MaxPool2d` in the CNN backbone with an equivalent-stride convolution, so
  the downsampling is learned instead of fixed to a max operation.
- **He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image
  Recognition.** (https://arxiv.org/abs/1512.03385)
  Why: source for `ResidualBlock`/`CRNNResidual` in
  `notebooks/07_crnn_architecture_variants.ipynb` — wraps each CNN stage in a residual
  (skip) connection while keeping the existing `MaxPool2d` downsampling unchanged, to
  isolate the residual-connection effect from the pooling/stride experiment above.
- **Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for
  Statistical Machine Translation.** (https://arxiv.org/abs/1406.1078); **Chung, J., Gulcehre,
  C., Cho, K., & Bengio, Y. (2014). Empirical Evaluation of Gated Recurrent Units on
  Sequence Modeling.** (https://arxiv.org/abs/1412.3555)
  Why: source for `CRNNGRU` in `notebooks/07_crnn_architecture_variants.ipynb` — swaps the
  BiLSTM for a BiGRU (fewer gates/parameters, comparable performance in the Chung et al.
  comparison), isolating the recurrent-cell-type effect from the CNN backbone.

- **Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). Connectionist
  Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural
  Networks.** ICML. (https://www.cs.toronto.edu/~graves/icml_2006.pdf)
  Why: source for CTC prefix beam search decoding, tried in
  `notebooks/07_crnn_architecture_variants.ipynb` as a decode-time-only alternative to
  `greedy_decode` on the already-trained `crnn_residual.pt` (0.720) — no retraining
  needed, since it only changes how output probabilities are turned into text.

## Comparable projects / official material

- **Hannun, A. (2017). Sequence Modeling With CTC.** Distill.
  (https://distill.pub/2017/ctc/)
  Why: practical write-up of the CTC prefix beam search algorithm (with worked
  pseudocode) used to implement `beam_search_decode` in
  `notebooks/07_crnn_architecture_variants.ipynb`, alongside the original Graves et
  al. 2006 paper above.
- **Loshchilov, I., & Hutter, F. (2016). SGDR: Stochastic Gradient Descent with Warm
  Restarts.** ICLR 2017. (https://arxiv.org/abs/1608.03983)
  Why: source for the cosine-annealing LR schedule tried on `CRNNResidual` in
  `notebooks/07_crnn_architecture_variants.ipynb` (no restarts, single cosine decay
  over `T_max=MAX_EPOCHS`) — chosen over `OneCycleLR` because it degrades gracefully
  under early stopping (monotonic decay, no need to complete a full up/down cycle),
  unlike OneCycle which assumes a fixed, known total step count.
- **Wigington, C., Stewart, S., Davis, B., Barrett, B., Price, B., & Cohen, S. (2017).
  Data Augmentation for Recognition of Handwritten Words and Lines Using a CNN-LSTM
  Network.** ICDAR. (https://ieeexplore.ieee.org/document/8270041)
  Why: source for the light geometric/intensity augmentation (rotation, brightness/
  contrast jitter, additive Gaussian noise) tried on `CRNNResidual`'s fit split in
  `notebooks/07_crnn_architecture_variants.ipynb` — directly about augmentation for
  CNN-LSTM handwritten line recognition, the same architecture family as this
  project's CRNN. Implemented with `torchvision.transforms.functional` (already
  installed) instead of adding Albumentations, to avoid repeating the
  opencv/numpy version conflict from the paddleocr/kraken install incident above.
- **Simard, P. Y., Steinkraus, D., & Platt, J. C. (2003). Best Practices for
  Convolutional Neural Networks Applied to Visual Document Analysis.** ICDAR.
  (https://www.microsoft.com/en-us/research/publication/best-practices-for-convolutional-neural-networks-applied-to-visual-document-analysis/)
  Why: source for elastic distortion, tried as an addition on top of the existing
  rotation/brightness/contrast/noise augmentation in
  `notebooks/07_crnn_architecture_variants.ipynb` (continuation from
  `crnn_residual_aug.pt`, 0.769) — a smooth random displacement field
  (Gaussian-smoothed noise, `torch.nn.functional.grid_sample`) simulating natural
  handwriting stroke variability, which affine transforms alone don't capture.
- **Amodei, D. et al. (2016). Deep Speech 2: End-to-End Speech Recognition in
  English and Mandarin.** ICML. (https://arxiv.org/abs/1512.02595)
  Why: source for the CTC beam search + n-gram language model shallow-fusion
  formula (`score = log P_ctc + alpha * log P_lm`) used in
  `notebooks/07_crnn_architecture_variants.ipynb`'s `beam_search_decode`, with a
  character n-gram LM trained on this project's own fit-split `Target` text
  (add-k smoothed counting, not KenLM — avoids a native-dependency install for a
  corpus this small, ~200K characters).
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

- **Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and
  Scalable Predictive Uncertainty Estimation using Deep Ensembles.** NeurIPS.
  (https://arxiv.org/abs/1612.01474)
  Why: source for the deep-ensemble approach in `notebooks/10_ensemble.ipynb`
  - train several copies of the winning `CRNNResidual` recipe from different
    random seeds and average their output probabilities before decoding,
    rather than ensembling architecturally different or much-weaker models.
- **Hu, E. J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language
  Models.** (https://arxiv.org/abs/2106.09685)
  Why: source for the LoRA adapters (via `peft`) trained on TrOCR's decoder
  attention projections in `notebooks/06_trocr_finetune.ipynb`, resumed
  2026-08-10 after full fine-tuning was dropped for destabilizing the
  pretrained weights. Result: plateaued at `score = 0.609`, still far
  below the CRNN ensemble — negative, final result for this project's
  TrOCR attempts (see README Progress).

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
- **PyTorch `torch.optim.lr_scheduler.ReduceLROnPlateau`**
  (https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)
  Why: added as an optional `scheduler` argument to
  `src/train.py::train_with_early_stopping`, tried in
  `notebooks/10_ensemble.ipynb` on the two new ensemble members — halves
  the LR only once the val score plateaus for a few epochs, reactive
  rather than the fixed preemptive decay of `CosineAnnealingLR` (tried and
  rejected in `notebooks/07`, see above).

## Models

- **Qwen2-VL-2B-Instruct** (https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
  Why: base model for the VLM approach in `reference/starter_pack/VLM/` (LoRA fine-tuned).
- **PaddleOCR** (https://github.com/PaddlePaddle/PaddleOCR)
  Why: detection + angle classification + recognition pipeline in `reference/starter_pack/Paddle-OCR/`.
- **Kraken** (https://kraken.re/)
  Why: OCR engine specialized for historical documents, used in `reference/starter_pack/Kraken-OCR/`.
