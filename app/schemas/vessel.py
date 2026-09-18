from pydantic import BaseModel, Field
from typing import List, Optional

class VesselSegmentationResponse(BaseModel):
    success: bool = True
    filename: str
    model: str = "DRIVE Vessel U-Net"
    dimensions: List[int]
    vessel_pixel_count: int
    retina_pixel_count: int
    vessel_density_percentage: float
    mask_base64_png: Optional[str] = None
    mask_image_url: Optional[str] = None
    disclaimer: str
