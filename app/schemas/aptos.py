from pydantic import BaseModel, Field
from typing import Dict, Optional, List

class ImageQualityInfo(BaseModel):
    status: str
    quality_score: float
    focus_score: float
    illumination_score: float
    field_of_view_score: float
    enhancement_applied: bool
    enhancements: List[str]
    recapture_required: bool
    recapture_feedback: str

class AptosPrediction(BaseModel):
    predicted_class: int = Field(..., ge=0, le=4)
    predicted_label: str
    class_description: str
    confidence: float
    class_probabilities: Dict[str, float]
    referable_probability: float
    referable_threshold: float
    referable: bool
    referable_label: str
    disclaimer: str

class AptosResponse(BaseModel):
    success: bool = True
    filename: str
    image_quality: ImageQualityInfo
    prediction: Optional[AptosPrediction] = None
    disclaimer: str
