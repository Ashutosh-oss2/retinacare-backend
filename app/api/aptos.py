from fastapi import APIRouter, File, UploadFile, HTTPException
from app.utils.image_processing import validate_and_decode_image
from app.services.aptos_service import run_aptos_inference
from app.schemas.aptos import AptosResponse
from app.utils.logging import logger

router = APIRouter(tags=["Model 1: APTOS DR Classification"])

async def _process_aptos_upload(file: UploadFile) -> AptosResponse:
    image_bgr, _, filename = await validate_and_decode_image(file)
    try:
        return run_aptos_inference(image_bgr, filename)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"APTOS inference error for {filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "inference_failed", "message": f"APTOS inference error: {str(e)}"}
        )

@router.post("/predict/aptos", response_model=AptosResponse)
@router.post("/api/predict/aptos", response_model=AptosResponse)
async def predict_aptos(file: UploadFile = File(..., description="Retinal fundus image (JPG, PNG, TIF)")):
    """Classifies diabetic retinopathy grade (0-4) with quality gating and referral triage."""
    return await _process_aptos_upload(file)

@router.post("/api/predict", response_model=AptosResponse)
async def legacy_predict(image: UploadFile = File(..., description="Legacy field name 'image'")):
    """Legacy endpoint alias for backward compatibility with frontend v1."""
    return await _process_aptos_upload(image)
