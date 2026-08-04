"""Image preprocessing, character-vocabulary building, and PyTorch dataset/
collate functions for the CRNN pipeline (Shi, Bai & Yao 2015 - see
RESOURCES.md). Validated in notebooks/05, 07, 08 before graduating here -
see the project workflow in the structuring-ml-projects skill.
"""

import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from src.data import build_image_path

TARGET_HEIGHT = 32
MAX_WIDTH = 1024


def build_vocab(texts: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Character vocabulary from a list of transcriptions.

    Index 0 is reserved for the CTC blank token (Graves et al. 2006), so
    real characters start at 1. Deterministic given the same `texts` -
    build from the fit split only, never val/test, or the vocabulary
    itself leaks split membership into the model.
    """
    chars = sorted(set("".join(texts)))
    char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
    idx_to_char = {i + 1: c for i, c in enumerate(chars)}
    return char_to_idx, idx_to_char


def load_and_resize_image(
    record_id: str, target_height: int = TARGET_HEIGHT, max_width: int = MAX_WIDTH
) -> torch.Tensor:
    """Grayscale, aspect-ratio-preserving resize to a fixed height, normalized to [-1, 1].

    Line crops vary widely in width/resolution across source documents
    (notebooks/02 EDA), so height is fixed and width scales with aspect
    ratio instead of a fixed crop.
    """
    img = Image.open(build_image_path(record_id)).convert("L")
    w, h = img.size
    new_w = max(8, min(max_width, round(w * target_height / h)))
    img = img.resize((new_w, target_height), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
    return torch.from_numpy(arr).unsqueeze(0)  # 1 x H x W


def ctc_output_length(width: int) -> int:
    """Time-step count CRNNResidual's two MaxPool2d(2, 2) stages produce for a given input width.

    Must match the model's actual downsampling, or CTCLoss/decoding index
    past the valid output - this formula is specific to CRNNResidual's
    pooling; a different architecture (e.g. stride-2 convs) needs its own
    (see notebooks/07's stride2_output_len note on why it isn't universal).
    """
    return (width // 2) // 2


class LineDataset(Dataset):
    """Wraps an ID(/Target) dataframe. Returns (image, label, id, text) -
    label is an empty tensor and text is None when no Target column is
    present (Test.csv has no ground truth)."""

    def __init__(self, df, char_to_idx: dict[str, int] | None = None):
        self.ids = df["ID"].tolist()
        self.targets = df["Target"].tolist() if "Target" in df.columns else None
        self.char_to_idx = char_to_idx

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        img = load_and_resize_image(self.ids[i])
        text = self.targets[i] if self.targets is not None else None
        if text is not None:
            label = torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)
        else:
            label = torch.tensor([], dtype=torch.long)
        return img, label, self.ids[i], text


def _gaussian_kernel1d(sigma: float, radius: int | None = None) -> torch.Tensor:
    if radius is None:
        radius = max(1, int(3 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _gaussian_blur_2d(field: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel = _gaussian_kernel1d(sigma)
    radius = kernel.numel() // 2
    field = F.conv2d(field, kernel.view(1, 1, 1, -1), padding=(0, radius))
    field = F.conv2d(field, kernel.view(1, 1, -1, 1), padding=(radius, 0))
    return field


def elastic_distort(img: torch.Tensor, alpha: float = 12.0, sigma: float = 4.0) -> torch.Tensor:
    """Simard, Steinkraus & Platt (2003) elastic distortion - see RESOURCES.md.

    Warps `img` (1 x H x W) via a random, Gaussian-smoothed displacement
    field (bilinear resample), simulating natural handwriting stroke
    variability. `alpha=0` reproduces the input exactly; output shape
    always matches input.
    """
    _, h, w = img.shape
    dx = _gaussian_blur_2d((torch.rand(1, 1, h, w) * 2 - 1), sigma) * alpha
    dy = _gaussian_blur_2d((torch.rand(1, 1, h, w) * 2 - 1), sigma) * alpha

    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij"
    )
    grid_x = (xx + dx[0, 0]) / max(w - 1, 1) * 2 - 1
    grid_y = (yy + dy[0, 0]) / max(h - 1, 1) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    warped = F.grid_sample(img.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True)
    return warped.squeeze(0)


class AugmentedLineDataset(Dataset):
    """Same as LineDataset but train-time-only augmented: light rotation,
    brightness/contrast jitter, additive Gaussian noise (Wigington et al.
    2017, ICDAR - see RESOURCES.md), plus elastic distortion (Simard et
    al. 2003 - see RESOURCES.md). This is the exact augmentation stack
    behind `crnn_residual_aug_elastic.pt` (0.720 -> 0.769 -> 0.796 on the
    fixed val sample, notebooks/07)."""

    def __init__(self, df, char_to_idx: dict[str, int]):
        self.ids = df["ID"].tolist()
        self.targets = df["Target"].tolist()
        self.char_to_idx = char_to_idx

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        img = load_and_resize_image(self.ids[i])

        angle = random.uniform(-3, 3)
        img = TF.rotate(img, angle, fill=-1.0)

        brightness = random.uniform(0.85, 1.15)
        contrast = random.uniform(0.85, 1.15)
        img = (img - img.mean()) * contrast + img.mean()
        img = (img * brightness).clamp(-1.0, 1.0)

        img = (img + torch.randn_like(img) * 0.03).clamp(-1.0, 1.0)
        img = elastic_distort(img, alpha=12.0, sigma=4.0).clamp(-1.0, 1.0)

        text = self.targets[i]
        label = torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)
        return img, label, self.ids[i], text


def collate_lines(batch):
    """Pads a batch of variable-width line images to the batch's max width;
    computes CTC input_lengths via ctc_output_length. `texts` entries are
    None for rows without a Target (Test.csv) - use `ids` to align
    predictions back to rows in that case."""
    imgs, labels, ids, texts = zip(*batch)
    widths = [im.shape[-1] for im in imgs]
    max_w = max(widths)
    padded = torch.zeros(len(imgs), 1, TARGET_HEIGHT, max_w)
    for i, im in enumerate(imgs):
        padded[i, :, :, : im.shape[-1]] = im
    input_lengths = torch.tensor([ctc_output_length(w) for w in widths], dtype=torch.long)
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    labels_cat = torch.cat(labels)
    return padded, labels_cat, input_lengths, target_lengths, list(ids), list(texts)
