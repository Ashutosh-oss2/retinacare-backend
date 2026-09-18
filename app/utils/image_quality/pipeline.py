"""Public quality-control pipeline."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import QualityConfig
from .enhancement import enhance_borderline
from .quality_assessment import QualityAssessment, QualityStatus, assess_quality
from .utils import ImageInput, load_image


@dataclass(frozen=True)
class QualityResult:
    assessment: QualityAssessment
    enhancement_applied: bool
    enhancements_used: tuple[str, ...]
    recapture_required: bool
    recapture_feedback: str
    processed_image: np.ndarray | None
    processed_image_path: str | None

    def to_dict(self) -> dict:
        """Return scalar metadata while retaining the processed image separately."""
        result = self.assessment.to_dict()
        result.update({
            "enhancement_applied": self.enhancement_applied,
            "enhancements_used": list(self.enhancements_used),
            "recapture_required": self.recapture_required,
            "recapture_feedback": self.recapture_feedback,
            "processed_image_path": self.processed_image_path,
            "processed_image": self.processed_image,
        })
        return result


def _feedback(assessment: QualityAssessment) -> str:
    reasons = []
    if assessment.focus_score < 0.52:
        reasons.append("hold the camera steady and refocus")
    if assessment.underexposure:
        reasons.append("increase illumination")
    if assessment.overexposure:
        reasons.append("reduce glare or illumination")
    if assessment.uneven_illumination:
        reasons.append("avoid reflections and center the light")
    if assessment.insufficient_field_of_view:
        reasons.append("move closer so the full retina is visible")
    if assessment.severely_cropped:
        reasons.append("recapture with the complete circular fundus in frame")
    return "; ".join(reasons).capitalize() + "." if reasons else "Recapture the image with the retina centered and in focus."


class QualityPipeline:
    """Assess and, only for borderline images, enhance a fundus image."""

    def __init__(self, config: QualityConfig | None = None):
        self.config = config or QualityConfig()

    def run(self, image: ImageInput, output_path: Optional[str | Path] = None) -> QualityResult:
        original = load_image(image)
        assessment = assess_quality(original, self.config)
        controlled = original
        used: list[str] = []
        if (assessment.quality_status == QualityStatus.BORDERLINE and
                assessment.quality_score >= self.config.enhancement_quality_floor and
                assessment.field_of_view_score >= self.config.enhancement_min_field_score and
                assessment.illumination_score >= self.config.enhancement_min_illumination_score):
            controlled, used = enhance_borderline(original, assessment, self.config)
            enhanced_assessment = assess_quality(controlled, self.config)
            improved = (enhanced_assessment.quality_score >=
                        assessment.quality_score + self.config.enhancement_min_quality_gain)
            if enhanced_assessment.quality_status == QualityStatus.ACCEPTABLE or improved:
                assessment = enhanced_assessment
            elif (assessment.focus_score == 0.0 and
                  enhanced_assessment.focus_score == 0.0):
                assessment = replace(enhanced_assessment, quality_status=QualityStatus.UNGRADABLE)
            else:
                assessment = enhanced_assessment
        recapture = assessment.quality_status == QualityStatus.UNGRADABLE
        saved_path = None
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), controlled):
                raise ValueError(f"Unable to write processed image: {output_path}")
            saved_path = str(output_path)
        return QualityResult(assessment, bool(used), tuple(used), recapture,
                             _feedback(assessment) if recapture else "", controlled, saved_path)


def assess_image_quality(image: ImageInput, config: QualityConfig | None = None) -> QualityAssessment:
    """Assess an image input without enhancement."""
    return assess_quality(load_image(image), config)


def process_image(image: ImageInput, output_path: Optional[str | Path] = None,
                  config: QualityConfig | None = None) -> QualityResult:
    """Run assessment and adaptive borderline enhancement."""
    return QualityPipeline(config).run(image, output_path)
