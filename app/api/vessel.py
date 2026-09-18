from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from app.utils.image_processing import validate_and_decode_image
from app.services.vessel_service import run_vessel_segmentation
from app.schemas.vessel import VesselSegmentationResponse
from app.utils.logging import logger

router = APIRouter(tags=["Model 2: DRIVE Vessel Segmentation"])

async def _process_vessel_upload(file: UploadFile, threshold: float = 0.5) -> VesselSegmentationResponse:
    image_bgr, _, filename = await validate_and_decode_image(file)
    try:
        return run_vessel_segmentation(image_bgr, filename, threshold=threshold, include_base64=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DRIVE vessel segmentation error for {filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "inference_failed", "message": f"Vessel segmentation error: {str(e)}"}
        )

@router.post("/predict/vessel", response_model=VesselSegmentationResponse)
@router.post("/api/predict/vessel", response_model=VesselSegmentationResponse)
@router.post("/predict/drive", response_model=VesselSegmentationResponse)
@router.post("/api/predict/drive", response_model=VesselSegmentationResponse)
async def predict_vessel(
    file: UploadFile = File(..., description="Retinal fundus image (JPG, PNG, TIF)"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Binary threshold for vessel detection")
):
    """Segments retinal vasculature and computes vascular density percentage."""
    return await _process_vessel_upload(file, threshold=threshold)
