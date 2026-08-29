"""Composition of the preprocessing stages into ``preprocess(image) -> tensor`` (R-8).

The stage sequence is data, not code: it comes from ``configs/data.yaml`` and is
validated at load time. That keeps the ordering question — which is a real one, see
D-008 — a configuration decision that can be measured, rather than something buried in
a function body.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from im2latex.config import PreprocessConfig
from im2latex.preprocessing import ops

#: Stage name -> operation. A name absent from this mapping is a configuration error.
STAGES: dict[str, Callable[..., np.ndarray]] = {
    "grayscale": ops.to_grayscale,
    "denoise": ops.denoise,
    "normalize_contrast": ops.normalize_contrast,
    "binarize": ops.binarize,
    "deskew": ops.deskew,
    "crop_to_content": ops.crop_to_content,
    "resize_and_pad": ops.resize_and_pad,
}


def apply_stages(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Run the configured stages in order, returning the final ``uint8`` array."""
    if image.size == 0:
        raise ValueError("Cannot preprocess an empty image")

    result = np.ascontiguousarray(image)
    for stage in config.stages:
        operation = STAGES[stage]  # validated by PreprocessConfig
        params: dict[str, Any] = dict(config.params.get(stage, {}))
        result = operation(result, **params)
    return result


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert a ``uint8`` array to a ``float32`` CHW tensor scaled to ``[0, 1]``."""
    normalized = image.astype(np.float32) / 255.0
    return torch.from_numpy(normalized).unsqueeze(0)


def preprocess(image: np.ndarray, config: PreprocessConfig) -> torch.Tensor:
    """Normalize a photograph or scan into model input.

    Returns a ``float32`` tensor of shape ``(1, height, width)`` with values in
    ``[0, 1]``, where 1 is ink — the inverse of the input's dark-ink-on-light-paper
    convention, established by :func:`im2latex.preprocessing.ops.binarize`.
    """
    return to_tensor(apply_stages(image, config))
