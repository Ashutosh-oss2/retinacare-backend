from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class LesionItem(BaseModel):
    name: str
    detected: bool
    pixel_count: int
    area_percentage: float

class IdridSegmentationResponse(BaseModel):
    success: bool = True
    filename: str
    dimensions: List[int]
    lesions: Dict[str, LesionItem]
    total_lesion_area: float
    clinical_risk_level: str
    clinical_action: str
    overlay_base64_png: Optional[str] = None
    overlay_image_url: Optional[str] = None
    disclaimer: str
