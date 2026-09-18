from pydantic import BaseModel
from typing import Optional
from app.schemas.aptos import AptosResponse
from app.schemas.vessel import VesselSegmentationResponse
from app.schemas.idrid import IdridSegmentationResponse

class UnifiedPredictionResponse(BaseModel):
    success: bool = True
    filename: str
    aptos: Optional[AptosResponse] = None
    vessel: Optional[VesselSegmentationResponse] = None
    idrid: Optional[IdridSegmentationResponse] = None
    disclaimer: str
