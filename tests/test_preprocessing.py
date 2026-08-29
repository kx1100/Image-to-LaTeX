"""Tests for the preprocessing operations and pipeline (R-8)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from conftest import INK, PAPER, apply_illumination_gradient, draw_expression
from im2latex.config import PreprocessConfig, load_config
from im2latex.preprocessing import apply_stages, ops, preprocess

REPO_CONFIG_STAGES = (
    "grayscale",
    "denoise",
    "normalize_contrast",
    "binarize",
    "deskew",
    "crop_to_content",
    "resize_and_pad",
)


def ink_fraction(binary: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of the given mask that the binary image marks as ink."""
    return float((binary[mask] > 0).sum()) / float(mask.sum())


# --------------------------------------------------------------------------- grayscale


def test_to_grayscale_reduces_colour_to_two_dimensions():
    colour = np.zeros((10, 12, 3), dtype=np.uint8)
    assert ops.to_grayscale(colour).shape == (10, 12)


def test_to_grayscale_handles_alpha():
    assert ops.to_grayscale(np.zeros((10, 12, 4), dtype=np.uint8)).shape == (10, 12)


def test_to_grayscale_passes_through_existing_grayscale():
    gray = np.full((10, 12), 128, dtype=np.uint8)
    assert np.array_equal(ops.to_grayscale(gray), gray)


def test_to_grayscale_rejects_unsupported_shape():
    with pytest.raises(ValueError, match="Cannot convert"):
        ops.to_grayscale(np.zeros((10, 12, 2), dtype=np.uint8))


# ----------------------------------------------------------------------------- denoise


def test_denoise_removes_salt_and_pepper_speckle(expression):
    clean, _ = expression
    rng = np.random.default_rng(0)
    noisy = clean.copy()
    speckle = rng.random(clean.shape) < 0.04
    noisy[speckle] = 0

    before = int(np.abs(noisy.astype(int) - clean.astype(int)).sum())
    after = int(np.abs(ops.denoise(noisy).astype(int) - clean.astype(int)).sum())
    assert after < before / 2


def test_denoise_rejects_even_kernel():
    with pytest.raises(ValueError, match="odd"):
        ops.denoise(np.zeros((10, 10), dtype=np.uint8), ksize=4)


def test_denoise_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown denoise method"):
        ops.denoise(np.zeros((10, 10), dtype=np.uint8), method="bilateral")


# ---------------------------------------------------------------------------- binarize


@pytest.mark.parametrize("method", ["adaptive", "otsu"])
def test_binarize_returns_ink_as_255_on_black(expression, method):
    image, mask = expression
    result = ops.binarize(image, method=method)

    assert set(np.unique(result).tolist()) <= {0, 255}
    assert ink_fraction(result, mask) > 0.9  # strokes recovered
    assert ink_fraction(result, ~mask) < 0.1  # paper not mistaken for ink


def test_binarize_rejects_even_block_size():
    with pytest.raises(ValueError, match="odd"):
        ops.binarize(np.zeros((20, 20), dtype=np.uint8), block_size=10)


def test_binarize_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown binarize method"):
        ops.binarize(np.zeros((20, 20), dtype=np.uint8), method="sauvola")


# ------------------------------------------------------------------ contrast ordering


@pytest.mark.parametrize("binarize_method", ["otsu", "adaptive"])
def test_contrast_normalization_before_binarization_survives_uneven_lighting(binarize_method):
    """The reason for the order in configs/data.yaml (D-008).

    Under a lighting gradient, thresholding first makes the shaded end of the page read
    as ink. Normalizing contrast first removes the gradient while leaving the ink/paper
    difference, so the threshold sees a page that is evenly lit.
    """
    clean, mask = draw_expression()
    lit_unevenly = apply_illumination_gradient(clean)

    params = {"binarize": {"method": binarize_method}}
    approved = apply_stages(
        lit_unevenly,
        PreprocessConfig(("grayscale", "denoise", "normalize_contrast", "binarize"), params),
    )
    contrast_last = apply_stages(
        lit_unevenly,
        PreprocessConfig(("grayscale", "denoise", "binarize", "normalize_contrast"), params),
    )

    # False ink on the shaded paper is the failure this ordering prevents.
    assert ink_fraction(approved, ~mask) < ink_fraction(contrast_last, ~mask)
    # ...and it is not bought by losing the strokes themselves.
    assert ink_fraction(approved, mask) > 0.85


def test_contrast_normalization_after_binarization_destroys_the_ink_convention(expression):
    """Why running it last is not merely worse but broken.

    Equalizing a two-valued histogram lifts the background off zero. Every operation
    after it locates ink with ``findNonZero``, so a non-zero background means the whole
    canvas reads as content and cropping and deskewing stop working.
    """
    image, _ = expression
    binary = ops.binarize(image, method="otsu")
    assert (binary == 0).any(), "the fixture must have a background to lose"

    equalized = ops.normalize_contrast(binary)
    assert not (equalized == 0).any(), "background was lifted off zero"

    # The concrete downstream consequence: cropping no longer crops anything.
    assert ops.crop_to_content(equalized, padding=0).shape == equalized.shape
    assert ops.crop_to_content(binary, padding=0).shape != binary.shape


# ------------------------------------------------------------------------------ deskew


@pytest.mark.parametrize("angle", [-8.0, -3.0, 3.0, 8.0])
def test_deskew_recovers_a_known_rotation(angle):
    clean, _ = draw_expression()
    height, width = clean.shape
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(clean, matrix, (width, height), borderValue=PAPER)

    binary = ops.binarize(rotated, method="otsu")
    residual = ops.estimate_skew_angle(ops.deskew(binary))
    assert abs(residual) < 2.0


def test_deskew_leaves_an_empty_image_alone():
    empty = np.zeros((40, 40), dtype=np.uint8)
    assert np.array_equal(ops.deskew(empty), empty)


def test_skew_estimate_is_declined_when_implausibly_large():
    """Beyond the configured limit the estimate is likelier noise than a tilted page."""
    clean, _ = draw_expression()
    binary = ops.binarize(clean, method="otsu")
    assert ops.estimate_skew_angle(binary, max_angle_deg=0.01) == 0.0


# ----------------------------------------------------------------------------- cropping


def test_crop_to_content_tightens_to_the_ink_plus_padding():
    image = np.zeros((100, 200), dtype=np.uint8)
    image[40:60, 80:120] = 255  # a 20x40 block of ink

    cropped = ops.crop_to_content(image, padding=5)
    assert cropped.shape == (20 + 10, 40 + 10)
    assert cropped[5:25, 5:45].all()
    assert not cropped[0:5, :].any()  # the re-added margin is background


def test_crop_to_content_without_padding_is_exactly_the_bounding_box():
    image = np.zeros((100, 200), dtype=np.uint8)
    image[40:60, 80:120] = 255
    assert ops.crop_to_content(image, padding=0).shape == (20, 40)


def test_crop_to_content_leaves_a_blank_image_alone():
    blank = np.zeros((30, 50), dtype=np.uint8)
    assert np.array_equal(ops.crop_to_content(blank), blank)


def test_crop_to_content_rejects_negative_padding():
    with pytest.raises(ValueError, match="non-negative"):
        ops.crop_to_content(np.zeros((10, 10), dtype=np.uint8), padding=-1)


# ------------------------------------------------------------------------------ resize


def test_resize_and_pad_produces_the_exact_target_canvas():
    image = np.full((50, 300), 255, dtype=np.uint8)
    assert ops.resize_and_pad(image, height=64, width=512).shape == (64, 512)


def test_resize_and_pad_preserves_aspect_ratio():
    image = np.full((100, 200), 255, dtype=np.uint8)  # 1:2
    result = ops.resize_and_pad(image, height=64, width=512)

    rows = np.flatnonzero(result.any(axis=1))
    columns = np.flatnonzero(result.any(axis=0))
    occupied_height = rows[-1] - rows[0] + 1
    occupied_width = columns[-1] - columns[0] + 1
    assert abs(occupied_width / occupied_height - 2.0) < 0.05


def test_resize_and_pad_does_not_overflow_a_wide_image():
    """A very wide expression is bounded by the canvas width, not the target height."""
    image = np.full((10, 4000), 255, dtype=np.uint8)
    result = ops.resize_and_pad(image, height=64, width=512)
    assert result.shape == (64, 512)
    assert result.any()


def test_resize_and_pad_rejects_a_non_positive_target():
    with pytest.raises(ValueError, match="positive"):
        ops.resize_and_pad(np.zeros((10, 10), dtype=np.uint8), height=0, width=10)


# ---------------------------------------------------------------------------- pipeline


def test_preprocess_returns_a_normalized_chw_tensor(expression):
    image, _ = expression
    config = load_config().preprocessing

    tensor = preprocess(image, config)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.shape == (1, 64, 512)
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_preprocess_marks_ink_as_high_values(expression):
    """Ink is 1 and paper is 0, inverting the input's dark-on-light convention."""
    image, _ = expression
    tensor = preprocess(image, load_config().preprocessing)

    assert float(tensor.max()) > 0.9  # ink is present and bright
    assert float(tensor.mean()) < 0.5  # most of the canvas is background


def test_preprocess_accepts_a_colour_photograph(expression):
    image, _ = expression
    colour = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    assert preprocess(colour, load_config().preprocessing).shape == (1, 64, 512)


def test_preprocess_handles_a_photograph_that_is_skewed_and_unevenly_lit():
    """The stages compose on the input R-8 was written for, not just on clean scans."""
    clean, _ = draw_expression()
    height, width = clean.shape
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 6.0, 1.0)
    photographed = apply_illumination_gradient(
        cv2.warpAffine(clean, matrix, (width, height), borderValue=PAPER)
    )

    tensor = preprocess(photographed, load_config().preprocessing)
    assert tensor.shape == (1, 64, 512)
    assert 0.01 < float(tensor.mean()) < 0.5  # ink recovered, page not flooded


def test_preprocess_rejects_an_empty_image():
    config = load_config().preprocessing
    with pytest.raises(ValueError, match="empty image"):
        preprocess(np.zeros((0, 0), dtype=np.uint8), config)


def test_repository_config_uses_the_approved_stage_order():
    """Guards the ordering decision itself against silent drift (D-008)."""
    assert load_config().preprocessing.stages == REPO_CONFIG_STAGES

    stages = load_config().preprocessing.stages
    assert stages.index("normalize_contrast") < stages.index("binarize")


def test_a_blank_page_survives_the_whole_pipeline():
    """No ink anywhere must not crash crop or resize."""
    blank = np.full((120, 300), PAPER, dtype=np.uint8)
    tensor = preprocess(blank, load_config().preprocessing)
    assert tensor.shape == (1, 64, 512)


def test_stages_are_applied_in_the_configured_order():
    """Reordering the config must actually reorder the work."""
    image, _ = draw_expression()
    crop_first = apply_stages(
        image,
        PreprocessConfig(
            ("grayscale", "binarize", "crop_to_content", "resize_and_pad"),
            {"crop_to_content": {"padding": 0}, "resize_and_pad": {"height": 32, "width": 64}},
        ),
    )
    resize_first = apply_stages(
        image,
        PreprocessConfig(
            ("grayscale", "binarize", "resize_and_pad", "crop_to_content"),
            {"crop_to_content": {"padding": 0}, "resize_and_pad": {"height": 32, "width": 64}},
        ),
    )
    assert crop_first.shape == (32, 64)
    assert resize_first.shape != crop_first.shape  # cropping after resizing trims padding


def test_ink_value_constants_are_what_the_fixtures_assume():
    assert INK < PAPER
