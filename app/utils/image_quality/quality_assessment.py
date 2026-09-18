"""Focus, illumination, field-of-view, and combined quality metrics."""

from dataclasses import asdict, dataclass
from enum import Enum

import cv2
import numpy as np

from .config import QualityConfig
from .utils import retinal_mask


class QualityStatus(str, Enum):
    ACCEPTABLE = "acceptable"
    BORDERLINE = "borderline"
    UNGRADABLE = "ungradable"


@dataclass(frozen=True)
class QualityAssessment:
    quality_score: float
    focus_score: float
    illumination_score: float
    field_of_view_score: float
    quality_status: QualityStatus
    focus_measure: float
    retinal_area_ratio: float
    underexposure: bool
    overexposure: bool
    uneven_illumination: bool
    insufficient_field_of_view: bool
    severely_cropped: bool

    def to_dict(self) -> dict:
        """Return JSON-friendly assessment values."""
        result = asdict(self)
        result["quality_status"] = self.quality_status.value
        return result


def _focus_score(image: np.ndarray, mask: np.ndarray, config: QualityConfig) -> tuple[float, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    values = laplacian[mask]
    measure = float(np.var(values)) if values.size else 0.0
    score = float(np.clip((measure - config.blur_laplacian_threshold) /
                          (config.sharp_laplacian_reference - config.blur_laplacian_threshold), 0.0, 1.0))
    return score, measure


def _illumination_score(image: np.ndarray, mask: np.ndarray, config: QualityConfig) -> tuple[float, bool, bool, bool]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    values = gray[mask].astype(np.float32)
    if values.size == 0:
        return 0.0, True, True, True
    mean = float(np.mean(values))
    under = mean < config.underexposure_mean_threshold or float(np.mean(values <= 5)) > config.clipped_pixel_ratio
    over = mean > config.overexposure_mean_threshold or float(np.mean(values >= 250)) > config.clipped_pixel_ratio
    blurred = cv2.GaussianBlur(gray, (0, 0), 9)
    regional_means = []
    row_bounds = np.linspace(0, gray.shape[0], 5, dtype=int)
    column_bounds = np.linspace(0, gray.shape[1], 5, dtype=int)
    for row_index in range(4):
        for column_index in range(4):
            row_start, row_end = row_bounds[row_index:row_index + 2]
            column_start, column_end = column_bounds[column_index:column_index + 2]
            region = blurred[row_start:row_end, column_start:column_end]
            region_mask = mask[row_start:row_end, column_start:column_end]
            if region_mask.any():
                regional_means.append(float(np.mean(region[region_mask])))
    regional_std = float(np.std(regional_means)) if regional_means else 0.0
    uneven = (float(np.std(values)) > config.illumination_std_threshold or
              regional_std > config.regional_illumination_std_threshold)
    penalty = float(under) * 0.42 + float(over) * 0.42 + float(uneven) * 0.28
    return float(np.clip(1.0 - penalty, 0.0, 1.0)), under, over, uneven


def _field_of_view_score(mask: np.ndarray, config: QualityConfig) -> tuple[float, float, bool, bool]:
    height, width = mask.shape
    area_ratio = float(np.mean(mask))
    if not np.any(mask):
        return 0.0, area_ratio, True, True
    ys, xs = np.where(mask)
    bbox_ratio = ((xs.max() - xs.min() + 1) / width) * ((ys.max() - ys.min() + 1) / height)
    touches = ((xs.min() == 0) + (ys.min() == 0) + (xs.max() == width - 1) + (ys.max() == height - 1))
    insufficient = area_ratio < config.min_retinal_area_ratio or bbox_ratio < config.min_retinal_bbox_ratio
    cropped = area_ratio < config.severe_crop_area_ratio or touches >= 2
    area_score = np.clip(area_ratio / config.min_retinal_area_ratio, 0.0, 1.0)
    bbox_score = np.clip(bbox_ratio / config.min_retinal_bbox_ratio, 0.0, 1.0)
    score = float(0.65 * area_score + 0.35 * bbox_score)
    return score, area_ratio, bool(insufficient), bool(cropped)


def assess_quality(image: np.ndarray, config: QualityConfig | None = None) -> QualityAssessment:
    """Assess a BGR image without changing it or applying model preprocessing."""
    config = config or QualityConfig()
    mask = retinal_mask(image, config)
    focus, measure = _focus_score(image, mask, config)
    illumination, under, over, uneven = _illumination_score(image, mask, config)
    field, area, insufficient, cropped = _field_of_view_score(mask, config)
    overall = float(np.clip(
        config.score_weight_focus * focus
        + config.score_weight_illumination * illumination
        + config.score_weight_field_of_view * field,
        0.0, 1.0
    ))
    if (cropped or area < config.severe_crop_area_ratio or under or over or
            (focus == 0.0 and illumination < 0.4)):
        status = QualityStatus.UNGRADABLE
    elif (overall >= config.acceptable_quality_score and not uneven and
          focus >= config.acceptable_focus_score and
          illumination >= config.acceptable_illumination_score and
          field >= config.acceptable_field_score):
        status = QualityStatus.ACCEPTABLE
    elif overall >= config.borderline_quality_score:
        status = QualityStatus.BORDERLINE
    else:
        status = QualityStatus.UNGRADABLE
    return QualityAssessment(overall, focus, illumination, field, status, measure, area,
                             under, over, uneven, insufficient, cropped)
