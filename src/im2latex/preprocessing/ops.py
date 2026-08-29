"""Individual preprocessing operations (R-8).

Each function is a pure transform on a NumPy array, so it can be tested and reordered
independently of the others. :mod:`im2latex.preprocessing.pipeline` composes them.

Ink convention
--------------
Input is a photograph or scan: *dark ink on light paper*. From :func:`binarize` onward
that is inverted, so **ink is 255 and background is 0**. Every downstream operation
depends on it: :func:`crop_to_content` finds content with ``findNonZero``, and
:func:`deskew` and :func:`resize_and_pad` fill new border area with 0, which is
therefore background rather than a black frame the model would read as ink.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "to_grayscale",
    "denoise",
    "normalize_contrast",
    "binarize",
    "deskew",
    "crop_to_content",
    "resize_and_pad",
]


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Reduce to a single 8-bit channel, passing an already-grayscale image through."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Cannot convert an image of shape {image.shape} to grayscale")


def denoise(image: np.ndarray, method: str = "median", ksize: int = 3) -> np.ndarray:
    """Suppress sensor and paper-texture noise.

    Median is the default: it removes the speckle that a phone camera puts on paper
    without softening stroke edges the way a Gaussian does, and thin strokes are
    exactly what the recognizer needs to keep.
    """
    if ksize % 2 == 0 or ksize < 1:
        raise ValueError(f"denoise ksize must be a positive odd integer, got {ksize}")
    if method == "median":
        return cv2.medianBlur(image, ksize)
    if method == "gaussian":
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    raise ValueError(f"Unknown denoise method: {method!r} (expected 'median' or 'gaussian')")


def normalize_contrast(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8
) -> np.ndarray:
    """Even out illumination with CLAHE.

    This must run *before* :func:`binarize` (D-008), for two reasons. A photograph of
    paper is rarely lit evenly — a hand shadow or a window on one side leaves a gradient
    that a threshold reads as ink — and equalizing locally first removes the gradient
    while leaving the ink/paper difference intact.

    Running it afterwards is not merely useless but actively destructive: CLAHE
    redistributes a two-valued histogram onto a non-zero floor (background 0 becomes 3),
    which erases the zero background that :func:`crop_to_content` and :func:`deskew`
    rely on to locate ink. Both would then treat the entire canvas as content.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(image)


def binarize(
    image: np.ndarray, method: str = "adaptive", block_size: int = 35, offset: float = 10.0
) -> np.ndarray:
    """Separate ink from paper, returning ink as 255 on a 0 background.

    ``adaptive`` thresholds against a local neighbourhood mean and is the default for
    photographs. ``otsu`` picks one global threshold from the histogram; it is cleaner
    on flatbed scans and is the case where the contrast-normalization ordering matters
    most, since a single threshold cannot adapt to a lighting gradient on its own.
    """
    if method == "otsu":
        _, result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return result
    if method == "adaptive":
        if block_size % 2 == 0 or block_size < 3:
            raise ValueError(f"binarize block_size must be an odd integer >= 3, got {block_size}")
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            offset,
        )
    raise ValueError(f"Unknown binarize method: {method!r} (expected 'adaptive' or 'otsu')")


def estimate_skew_angle(image: np.ndarray, max_angle_deg: float = 15.0) -> float:
    """Estimate the rotation of an ink-on-black image, in degrees.

    Hough is tried first, since the baseline of an expression and the bars of any
    fractions give strong near-horizontal evidence. A single short expression may not
    produce enough long segments for Hough to be reliable, so the fallback is the
    minimum-area rectangle around the ink, which needs no line structure at all.

    Returns 0.0 when there is no ink, or when the estimate exceeds ``max_angle_deg`` —
    beyond that the estimate is more likely a misdetection than a genuinely tilted
    photograph, and rotating on it would make the image worse.
    """
    points = cv2.findNonZero(image)
    if points is None:
        return 0.0

    angle: float | None = None
    lines = cv2.HoughLinesP(
        image,
        rho=1,
        theta=np.pi / 180.0,
        threshold=80,
        minLineLength=max(20, image.shape[1] // 8),
        maxLineGap=10,
    )
    if lines is not None:
        candidates = []
        # OpenCV has returned this as both (N, 1, 4) and (N, 4) across versions.
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            degrees = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1)))
            # Fold onto the near-horizontal range; vertical strokes carry no skew
            # information and would otherwise dominate the median.
            if -max_angle_deg <= degrees <= max_angle_deg:
                candidates.append(degrees)
        if candidates:
            angle = float(np.median(candidates))

    if angle is None:
        rect_angle = cv2.minAreaRect(points)[-1]
        # OpenCV reports the angle in (0, 90]; fold it to the nearest axis.
        angle = float(rect_angle - 90.0 if rect_angle > 45.0 else rect_angle)

    return 0.0 if abs(angle) > max_angle_deg else angle


def deskew(image: np.ndarray, max_angle_deg: float = 15.0) -> np.ndarray:
    """Rotate an ink-on-black image back to horizontal."""
    angle = estimate_skew_angle(image, max_angle_deg=max_angle_deg)
    if angle == 0.0:
        return image

    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,  # keeps the image strictly two-valued
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,  # background, per the ink convention
    )


def crop_to_content(image: np.ndarray, padding: int = 8) -> np.ndarray:
    """Crop to the ink bounding box, then re-add a uniform margin.

    The margin is added back deliberately: a stroke flush against the image edge is
    ambiguous about whether it was cut off, and the encoder should not have to guess.
    An image with no ink is returned unchanged rather than collapsed to nothing.
    """
    if padding < 0:
        raise ValueError(f"crop_to_content padding must be non-negative, got {padding}")

    points = cv2.findNonZero(image)
    if points is None:
        return image

    x, y, width, height = cv2.boundingRect(points)
    cropped = image[y : y + height, x : x + width]
    if padding == 0:
        return cropped
    return cv2.copyMakeBorder(
        cropped, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0
    )


def resize_and_pad(image: np.ndarray, height: int = 64, width: int = 512) -> np.ndarray:
    """Scale to a fixed canvas, preserving aspect ratio.

    Aspect ratio is preserved because it carries meaning: the ratio of a fraction bar to
    its numerator, or the relative size of a superscript, is part of what distinguishes
    one expression from another. Content is left-aligned and vertically centred, and the
    remaining canvas is background.

    Interpolation makes the result continuous-valued rather than strictly binary; that
    is intended, since anti-aliased strokes carry more information to the encoder than
    hard-thresholded ones.
    """
    if height < 1 or width < 1:
        raise ValueError(f"resize_and_pad target must be positive, got {height}x{width}")

    source_height, source_width = image.shape[:2]
    if source_height == 0 or source_width == 0:
        return np.zeros((height, width), dtype=np.uint8)

    scale = min(height / source_height, width / source_width)
    new_height = max(1, min(height, int(round(source_height * scale))))
    new_width = max(1, min(width, int(round(source_width * scale))))

    # INTER_AREA is the correct choice when shrinking, which is the usual direction
    # from a phone photograph; INTER_LINEAR is better when enlarging a small crop.
    shrinking = new_height < source_height or new_width < source_width
    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR,
    )

    canvas = np.zeros((height, width), dtype=image.dtype)
    top = (height - new_height) // 2
    canvas[top : top + new_height, 0:new_width] = resized
    return canvas
