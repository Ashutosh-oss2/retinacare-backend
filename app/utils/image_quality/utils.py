"""Image loading, validation, and retinal-region utilities."""

from pathlib import Path
from typing import Union

import cv2
import numpy as np

ImageInput = Union[str, Path, np.ndarray]


def load_image(image: ImageInput) -> np.ndarray:
    """Load an image as a BGR uint8 array, or validate an array input."""
    if isinstance(image, (str, Path)):
        loaded = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if loaded is None:
            raise ValueError(f"Unable to read image: {image}")
        return loaded
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty path or NumPy array")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("image must have shape (height, width, 3) or (height, width, 4)")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image.copy()


def retinal_mask(image: np.ndarray, config) -> np.ndarray:
    """Return the largest plausible non-black fundus region."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray > config.background_gray_threshold).astype(np.uint8) * 255
    kernel_size = max(3, min(image.shape[:2]) // 100 * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if components <= 1:
        return mask.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def masked_values(channel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Extract finite values inside a mask, returning an empty array if needed."""
    values = channel[mask]
    return values[np.isfinite(values)]
