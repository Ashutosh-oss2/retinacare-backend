"""Synthetic regression tests for the independent quality module."""

import cv2
import numpy as np

from .config import QualityConfig
from .pipeline import process_image
from .quality_assessment import QualityStatus, assess_quality


def fundus(radius=105, value=105, noise=12):
    rng = np.random.default_rng(7)
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:256, :256]
    mask = (xx - 128) ** 2 + (yy - 128) ** 2 <= radius ** 2
    texture = np.clip(value + rng.normal(0, noise, mask.sum()), 0, 255).astype(np.uint8)
    for channel in range(3):
        image[:, :, channel][mask] = texture
    cv2.circle(image, (128, 128), radius // 3, (145, 75, 55), 2)
    return image


def test_good_fundus_is_acceptable():
    assessment = assess_quality(fundus())
    assert assessment.quality_status == QualityStatus.ACCEPTABLE
    assert assessment.quality_score >= 0.70


def test_blurry_image_has_low_focus():
    assessment = assess_quality(cv2.GaussianBlur(fundus(noise=4), (31, 31), 0))
    assert assessment.focus_score < 0.52
    assert assessment.focus_measure < 180


def test_dark_image_is_ungradable():
    assessment = assess_quality(fundus(value=8, noise=0))
    assert assessment.underexposure
    assert assessment.quality_status == QualityStatus.UNGRADABLE


def test_overexposed_image_is_ungradable():
    assessment = assess_quality(fundus(value=250, noise=0))
    assert assessment.overexposure
    assert assessment.quality_status == QualityStatus.UNGRADABLE


def test_uneven_illumination_is_detected():
    image = fundus(value=90, noise=8)
    yy, xx = np.ogrid[:256, :256]
    fundus_mask = (xx - 128) ** 2 + (yy - 128) ** 2 <= 105 ** 2
    gradient = np.tile(np.linspace(0, 150, 256, dtype=np.uint8), (256, 1))
    image[fundus_mask] = np.clip(image[fundus_mask].astype(np.int16) + gradient[fundus_mask, None], 0, 255)
    assessment = assess_quality(image)
    assert assessment.uneven_illumination
    assert assessment.illumination_score < 1.0


def test_insufficient_field_of_view_is_ungradable():
    assessment = assess_quality(fundus(radius=55))
    assert assessment.insufficient_field_of_view
    assert assessment.quality_status == QualityStatus.UNGRADABLE


def test_borderline_image_is_enhanced_adaptively():
    image = fundus(value=75, noise=9)
    yy, xx = np.ogrid[:256, :256]
    fundus_mask = (xx - 128) ** 2 + (yy - 128) ** 2 <= 105 ** 2
    gradient = np.tile(np.linspace(0, 100, 256, dtype=np.uint8), (256, 1))
    image[fundus_mask] = np.clip(image[fundus_mask].astype(np.int16) + gradient[fundus_mask, None], 0, 255)
    result = process_image(image)
    assert result.assessment.quality_status == QualityStatus.ACCEPTABLE
    assert result.enhancement_applied
    assert "illumination_normalization" in result.enhancements_used
    assert result.processed_image is not None


def test_zero_focus_score_does_not_block_enhancement_attempt():
    config = QualityConfig(
        blur_laplacian_threshold=100000.0,
        sharp_laplacian_reference=100001.0,
    )
    result = process_image(fundus(), config=config)
    assert result.assessment.focus_score == 0.0
    assert result.enhancement_applied
    assert "denoising" in result.enhancements_used


def test_enhancement_is_followed_by_reassessment():
    image = fundus(value=75, noise=9)
    yy, xx = np.ogrid[:256, :256]
    fundus_mask = (xx - 128) ** 2 + (yy - 128) ** 2 <= 105 ** 2
    gradient = np.tile(np.linspace(0, 100, 256, dtype=np.uint8), (256, 1))
    image[fundus_mask] = np.clip(image[fundus_mask].astype(np.int16) + gradient[fundus_mask, None], 0, 255)
    result = process_image(image)
    reassessed = assess_quality(result.processed_image)
    assert result.enhancement_applied
    assert result.assessment == reassessed


def test_ungradable_image_is_rejected_without_enhancement():
    result = process_image(cv2.GaussianBlur(fundus(value=8, noise=0), (31, 31), 0))
    assert result.recapture_required
    assert not result.enhancement_applied
    assert result.recapture_feedback
