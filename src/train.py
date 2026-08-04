"""End-to-end: load -> vocab/features -> fit CRNNResidual -> evaluate -> save.

Reproduces the project's best pipeline: CRNN with residual blocks (He et
al. 2015) trained with dropout + AdamW weight decay + early stopping
(Srivastava et al. 2014; Loshchilov & Hutter 2019; Prechelt 1998), data
augmentation + elastic distortion (Wigington et al. 2017; Simard,
Steinkraus & Platt 2003), decoded with CTC beam search + a 4-gram
character LM (Graves et al. 2006; Amodei et al. 2016) - see RESOURCES.md
for every citation, and notebooks/05, 07, 08 for the validated
progression. Reference score: 0.826 on the fixed 120-row val sample,
0.82658842 on the Zindi leaderboard.

`main()` is a multi-hour GPU training run - per this project's practice
(see README), run it yourself rather than via an automated tool; it isn't
invoked as a side effect of importing this module.
"""

import copy
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import ID_COLUMN, MODELS_DIR, RANDOM_STATE, TARGET_COLUMN
from src.data import load_raw_data, save_output, split_train_val
from src.evaluate import combined_score
from src.features import AugmentedLineDataset, LineDataset, build_vocab, collate_lines
from src.model import CRNNResidual, CharNgramLM, greedy_decode, predict

DROPOUT = 0.2
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 8
BATCH_SIZE = 32
BLANK = 0
LM_N = 4
LM_K = 1e-3
BEAM_WIDTH = 10
LM_ALPHA = 0.35
CHECKPOINT_NAME = "crnn_residual_aug_elastic.pt"


def _evaluate_greedy(model, loader, idx_to_char, device) -> dict:
    model.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for imgs, _targets_cat, input_lengths, _target_lengths, _ids, texts in loader:
            imgs = imgs.to(device)
            log_probs = model(imgs).cpu()
            hyps.extend(greedy_decode(log_probs, input_lengths, idx_to_char, blank=BLANK))
            refs.extend(texts)
    model.train()
    return combined_score(refs, hyps)


def train_with_early_stopping(
    model, optimizer, fit_dl, val_dl, idx_to_char, device, max_epochs, patience,
    initial_best_score=-1.0, scheduler=None,
):
    """Trains until `max_epochs` or `patience` epochs pass with no val-score
    improvement (Prechelt 1998, see RESOURCES.md); returns the best
    checkpoint. `initial_best_score` lets a continuation run only count an
    epoch as an improvement if it beats an already-known reference (e.g. a
    loaded checkpoint's score).

    `scheduler`, if given, must be a `ReduceLROnPlateau` (or anything else
    whose `.step()` takes the epoch's score) - called once per epoch with
    `score["score"]`. Reactive, not preemptive: only lowers the LR once the
    model has genuinely stopped improving for a few epochs, unlike
    `CosineAnnealingLR` (tried and rejected in notebooks/07 - it decayed LR
    on a fixed schedule regardless of whether the model was still
    improving, and lost to constant LR)."""
    ctc_loss = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    t0 = time.time()
    history = []
    best_score = initial_best_score
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_since_improvement = 0

    for epoch in range(1, max_epochs + 1):
        total_loss, n = 0.0, 0
        for imgs, targets_cat, input_lengths, target_lengths, _ids, _texts in fit_dl:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            log_probs = model(imgs)
            loss = ctc_loss(log_probs.permute(1, 0, 2), targets_cat, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
        train_loss = total_loss / n

        score = _evaluate_greedy(model, val_dl, idx_to_char, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **score})
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:2d} | lr {current_lr:.2e} | loss {train_loss:.3f} | val CER {score['cer_weighted']:.3f} "
            f"WER {score['wer_weighted']:.3f} score {score['score']:.3f} | {time.time() - t0:.0f}s"
        )

        if scheduler is not None:
            scheduler.step(score["score"])

        if score["score"] > best_score:
            best_score, best_epoch = score["score"], epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    return best_state, best_score, best_epoch, history


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df = load_raw_data("train")
    fit_df, val_df = split_train_val(train_df)
    val_sample = val_df.sample(120, random_state=RANDOM_STATE).reset_index(drop=True)

    char_to_idx, idx_to_char = build_vocab(fit_df[TARGET_COLUMN].tolist())
    num_classes = len(char_to_idx) + 1

    fit_ds = AugmentedLineDataset(fit_df, char_to_idx)
    val_ds = LineDataset(val_sample, char_to_idx)
    fit_dl = DataLoader(fit_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_lines)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_lines)

    model = CRNNResidual(num_classes, dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)

    best_state, best_score, best_epoch, _history = train_with_early_stopping(
        model, optimizer, fit_dl, val_dl, idx_to_char, device, max_epochs=MAX_EPOCHS, patience=PATIENCE
    )
    model.load_state_dict(best_state)
    print(f"Best epoch {best_epoch}, greedy val score {best_score:.4f}")

    lm = CharNgramLM(fit_df[TARGET_COLUMN].tolist(), n=LM_N, k=LM_K)
    _ids, hyps, refs = predict(
        model, val_dl, idx_to_char, device, blank=BLANK, beam_width=BEAM_WIDTH, lm=lm, alpha=LM_ALPHA
    )
    final_score = combined_score(refs, hyps)
    print(f"beam+LM val score: {final_score['score']:.4f} (reference: 0.826)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "char_to_idx": char_to_idx, "idx_to_char": idx_to_char},
        MODELS_DIR / CHECKPOINT_NAME,
    )


def generate_submission(checkpoint_path=None, output_filename: str = "submission.csv"):
    """Inference-only entry point: loads an already-trained checkpoint and
    decodes Test.csv with the project's default beam+LM pipeline - no
    retraining. Mirrors the Zindi submission generated in
    notebooks/08_decoding_strategies.ipynb (leaderboard score
    0.82658842)."""
    checkpoint_path = checkpoint_path or (MODELS_DIR / CHECKPOINT_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    char_to_idx, idx_to_char = checkpoint["char_to_idx"], checkpoint["idx_to_char"]
    num_classes = len(char_to_idx) + 1

    model = CRNNResidual(num_classes, dropout=DROPOUT).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_df = load_raw_data("train")
    fit_df, _val_df = split_train_val(train_df)
    lm = CharNgramLM(fit_df[TARGET_COLUMN].tolist(), n=LM_N, k=LM_K)

    test_df = load_raw_data("test")
    test_ds = LineDataset(test_df, char_to_idx=None)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_lines)

    ids, hyps, _refs = predict(
        model, test_dl, idx_to_char, device, blank=BLANK, beam_width=BEAM_WIDTH, lm=lm, alpha=LM_ALPHA
    )
    submission = pd.DataFrame({ID_COLUMN: ids, TARGET_COLUMN: hyps})
    return save_output(submission, output_filename)


if __name__ == "__main__":
    main()
