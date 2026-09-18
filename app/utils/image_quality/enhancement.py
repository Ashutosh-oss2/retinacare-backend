"""Adaptive enhancement for borderline fundus images."""

import cv2
import numpy as np

from .config import QualityConfig
from .quality_assessment import QualityAssessment


def _illumination_normalization(image: np.ndarray, config: QualityConfig) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0].astype(np.float32)
    kernel = max(3, config.illumination_blur_kernel | 1)
    background = cv2.GaussianBlur(lightness, (kernel, kernel), 0)
    corrected = np.clip(lightness - background + np.mean(background), 0, 255).astype(np.uint8)
    lab[:, :, 0] = corrected
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_borderline(image: np.ndarray, assessment: QualityAssessment,
                       config: QualityConfig | None = None) -> tuple[np.ndarray, list[str]]:
    """Apply only the enhancements indicated by the assessment."""
    config = config or QualityConfig()
    enhanced = image.copy()
    used: list[str] = []
    if assessment.underexposure or assessment.overexposure or assessment.uneven_illumination:
        enhanced = _illumination_normalization(enhanced, config)
        used.append("illumination_normalization")
    if assessment.uneven_illumination:
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(config.clahe_clip_limit, (config.clahe_tile_grid_size,
                                                           config.clahe_tile_grid_size))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        used.append("clahe")
    if assessment.focus_score < config.acceptable_focus_score:
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, config.denoise_strength,
                                                    config.denoise_strength, 7, 21)
        used.append("denoising")
    return enhanced, used
