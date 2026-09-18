from pydantic import BaseModel, Field
from typing import Dict, Any, List

class ModelsStatus(BaseModel):
    model_1_aptos: bool = Field(..., description="APTOS DR Classifier status")
    model_2_drive: bool = Field(..., description="DRIVE Vessel Segmentation status")
    model_3_idrid: bool = Field(..., description="IDRiD Lesion Segmentation status")

class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    models: ModelsStatus
    uptime_seconds: float = Field(..., example=12.4)
    version: str = Field(..., example="1.0.0")
    disclaimer: str

class EndpointInfo(BaseModel):
    method: str
    path: str
    desc: str

class RootServiceResponse(BaseModel):
    status: str = "online"
    service: str = "RetinaCare Triple AI Diagnostic Suite Backend"
    version: str = "1.0.0"
    endpoints: List[EndpointInfo]
    disclaimer: str
