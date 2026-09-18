from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from app.utils.image_processing import validate_and_decode_image
from app.services.aptos_service import run_aptos_inference
from app.services.vessel_service import run_vessel_segmentation
from app.services.idrid_service import run_idrid_segmentation
from app.schemas.unified import UnifiedPredictionResponse
from app.utils.logging import logger

router = APIRouter(tags=["Unified Multi-Model Analysis"])

CLINICAL_DISCLAIMER = (
    "This system provides AI-assisted retinal screening and segmentation support "
    "and is not a substitute for professional medical diagnosis."
)

@router.post("/predict/all", response_model=UnifiedPredictionResponse)
@router.post("/api/predict/all", response_model=UnifiedPredictionResponse)
async def predict_all(
    file: UploadFile = File(..., description="Retinal fundus image (JPG, PNG, TIF)"),
    vessel_threshold: float = Query(0.5, ge=0.0, le=1.0),
    lesion_threshold: float = Query(0.5, ge=0.0, le=1.0)
):
    """
    Executes the complete triple-model inference pipeline on a single uploaded fundus image:
    1. Model 1 (APTOS DR Classification)
    2. Model 2 (DRIVE Vessel Segmentation)
    3. Model 3 (IDRiD Lesion Segmentation)
    """
    image_bgr, _, filename = await validate_and_decode_image(file)

    try:
        aptos_res = run_aptos_inference(image_bgr, filename)
        vessel_res = run_vessel_segmentation(image_bgr, filename, threshold=vessel_threshold, include_base64=True)
        idrid_res = run_idrid_segmentation(image_bgr, filename, threshold=lesion_threshold, include_base64=True)

        return UnifiedPredictionResponse(
            success=True,
            filename=filename,
            aptos=aptos_res,
            vessel=vessel_res,
            idrid=idrid_res,
            disclaimer=CLINICAL_DISCLAIMER
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unified prediction error for {filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "unified_inference_failed", "message": f"Multi-model analysis error: {str(e)}"}
        )
