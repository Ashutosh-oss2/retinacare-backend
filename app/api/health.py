import time
from fastapi import APIRouter
from app.services.model_registry import registry
from app.schemas.health import HealthResponse, RootServiceResponse, ModelsStatus, EndpointInfo

router = APIRouter(tags=["Health & Service Discovery"])
_START_TIME = time.time()
CLINICAL_DISCLAIMER = (
    "This system provides AI-assisted retinal screening and segmentation support "
    "and is not a substitute for professional medical diagnosis."
)

app_root_endpoints = [
    EndpointInfo(method="GET", path="/health", desc="Service health and model load state"),
    EndpointInfo(method="GET", path="/api/health", desc="API health status alias"),
    EndpointInfo(method="GET", path="/api/model-info", desc="AI Model architectures and metadata"),
    EndpointInfo(method="GET", path="/api/model-stats", desc="Audited quantitative benchmarks"),
    EndpointInfo(method="POST", path="/predict/aptos", desc="Model 1: APTOS DR Classification"),
    EndpointInfo(method="POST", path="/predict/vessel", desc="Model 2: DRIVE Vessel Segmentation"),
    EndpointInfo(method="POST", path="/predict/idrid", desc="Model 3: IDRiD Lesion Segmentation"),
    EndpointInfo(method="POST", path="/predict/all", desc="Unified Multi-Model Analysis"),
]

@router.get("/", response_model=RootServiceResponse)
def root_discovery():
    return RootServiceResponse(
        status="online",
        service="RetinaCare Triple AI Diagnostic Suite Backend",
        version="1.0.0",
        endpoints=app_root_endpoints,
        disclaimer=CLINICAL_DISCLAIMER
    )

@router.get("/health", response_model=HealthResponse)
@router.get("/api/health", response_model=HealthResponse)
def health_check():
    models_status = ModelsStatus(
        model_1_aptos=registry.is_aptos_loaded,
        model_2_drive=registry.is_vessel_loaded,
        model_3_idrid=registry.is_idrid_loaded,
    )
    overall_status = "healthy" if all([
        registry.is_aptos_loaded,
        registry.is_vessel_loaded,
        registry.is_idrid_loaded
    ]) else "degraded"

    return HealthResponse(
        status=overall_status,
        models=models_status,
        uptime_seconds=round(time.time() - _START_TIME, 2),
        version="1.0.0",
        disclaimer=CLINICAL_DISCLAIMER
    )
