"""Configuration for image quality assessment.

Thresholds are intentionally centralized here so deployment-specific camera
characteristics can be calibrated without changing the assessment code.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityConfig:
    background_gray_threshold: int = 12
    min_retinal_area_ratio: float = 0.20
    min_retinal_bbox_ratio: float = 0.45
    max_border_touch_ratio: float = 0.90
    severe_crop_area_ratio: float = 0.32
    min_retinal_pixels: int = 500
    blur_laplacian_threshold: float = 35.0
    sharp_laplacian_reference: float = 180.0
    underexposure_mean_threshold: float = 38.0
    overexposure_mean_threshold: float = 220.0
    clipped_pixel_ratio: float = 0.08
    illumination_std_threshold: float = 58.0
    regional_illumination_std_threshold: float = 22.0
    acceptable_focus_score: float = 0.52
    acceptable_illumination_score: float = 0.62
    acceptable_field_score: float = 0.65
    acceptable_quality_score: float = 0.70
    borderline_quality_score: float = 0.45
    enhancement_quality_floor: float = 0.45
    enhancement_min_field_score: float = 0.65
    enhancement_min_illumination_score: float = 0.35
    enhancement_min_quality_gain: float = 0.02
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    denoise_strength: int = 5
    illumination_blur_kernel: int = 51
    score_weight_focus: float = 0.35
    score_weight_illumination: float = 0.35
    score_weight_field_of_view: float = 0.30
