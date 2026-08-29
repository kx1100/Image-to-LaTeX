"""Shared fixtures.

Test images are drawn programmatically rather than committed as files: the tests then
have no binary fixtures to maintain, no network dependency, and no dependency on the
synthetic renderer, which is a later slice of M1.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

PAPER = 245  # light paper, not pure white - a photograph never is
INK = 30  # dark stroke, not pure black


def draw_expression(height: int = 140, width: int = 420) -> tuple[np.ndarray, np.ndarray]:
    """Draw a crude 'a/b + x' on paper.

    Returns the grayscale image and the boolean ink mask that produced it, so tests can
    measure what a pipeline recovered against what was actually drawn.
    """
    image = np.full((height, width), PAPER, dtype=np.uint8)

    cv2.line(image, (40, 40), (40, 100), INK, 3)  # a vertical stroke
    cv2.line(image, (100, 70), (170, 70), INK, 4)  # a fraction bar
    cv2.line(image, (120, 30), (150, 55), INK, 3)  # numerator
    cv2.line(image, (120, 85), (150, 110), INK, 3)  # denominator
    cv2.line(image, (220, 70), (270, 70), INK, 3)  # plus, horizontal
    cv2.line(image, (245, 45), (245, 95), INK, 3)  # plus, vertical
    cv2.line(image, (320, 45), (360, 95), INK, 3)  # x, one stroke
    cv2.line(image, (360, 45), (320, 95), INK, 3)  # x, other stroke

    return image, image < (PAPER - 40)


def apply_illumination_gradient(image: np.ndarray, darkest: float = 0.30) -> np.ndarray:
    """Darken the image progressively left to right.

    Stands in for the shadow a hand or a window leaves across a photographed page - the
    condition R-8's contrast normalization exists to survive.
    """
    width = image.shape[1]
    ramp = np.linspace(1.0, darkest, width, dtype=np.float32)
    return np.clip(image.astype(np.float32) * ramp[None, :], 0, 255).astype(np.uint8)


@pytest.fixture
def expression() -> tuple[np.ndarray, np.ndarray]:
    """A clean, evenly lit expression and its ink mask."""
    return draw_expression()
