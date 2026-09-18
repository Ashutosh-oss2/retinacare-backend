from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from app.utils.image_processing import validate_and_decode_image
from app.services.idrid_service import run_idrid_segmentation
from app.schemas.idrid import IdridSegmentationResponse
from app.utils.logging import logger

router = APIRouter(tags=["Model 3: IDRiD Lesion Segmentation"])

async def _process_idrid_upload(file: UploadFile, threshold: float = 0.5) -> IdridSegmentationResponse:
    image_bgr, _, filename = await validate_and_decode_image(file)
    try:
        return run_idrid_segmentation(image_bgr, filename, threshold=threshold, include_base64=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"IDRiD lesion segmentation error for {filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "inference_failed", "message": f"Lesion segmentation error: {str(e)}"}
        )

@router.post("/predict/idrid", response_model=IdridSegmentationResponse)
@router.post("/api/predict/idrid", response_model=IdridSegmentationResponse)
async def predict_idrid(
    file: UploadFile = File(..., description="Retinal fundus image (JPG, PNG, TIF)"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Binary threshold for lesion detection")
):
    """Segments 4 retinal lesion classes (MA, HE, EX, SE) and generates a diagnostic color-coded overlay."""
    return await _process_idrid_upload(file, threshold=threshold)
