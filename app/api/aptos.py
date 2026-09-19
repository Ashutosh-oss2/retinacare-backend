from fastapi import APIRouter, File, UploadFile, HTTPException
from app.utils.image_processing import validate_and_decode_image
from app.services.aptos_service import run_aptos_inference
from app.schemas.aptos import AptosResponse
from app.utils.logging import logger

router = APIRouter(tags=["Model 1: APTOS DR Classification"])

import time
import gc

async def _process_aptos_upload(file: UploadFile) -> AptosResponse:
    start_time = time.time()
    image_bgr, _, filename = await validate_and_decode_image(file)
    logger.info(f"APTOS request received for '{filename}' ({image_bgr.shape[1]}x{image_bgr.shape[0]} px)")
    try:
        response = run_aptos_inference(image_bgr, filename)
        duration_ms = round((time.time() - start_time) * 1000, 1)
        if response.prediction:
            logger.info(
                f"APTOS inference success for '{filename}' in {duration_ms}ms: "
                f"{response.prediction.predicted_label} ({response.prediction.confidence}%) - {response.prediction.referable_label}"
            )
        else:
            logger.info(
                f"APTOS quality triage for '{filename}' in {duration_ms}ms: "
                f"Recapture required ({response.image_quality.status})"
            )
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"APTOS inference error for {filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "inference_failed", "message": f"APTOS inference error: {str(e)}"}
        )
    finally:
        del image_bgr
        gc.collect()

@router.post("/predict/aptos", response_model=AptosResponse)
@router.post("/api/predict/aptos", response_model=AptosResponse)
async def predict_aptos(file: UploadFile = File(..., description="Retinal fundus image (JPG, PNG, TIF)")):
    """Classifies diabetic retinopathy grade (0-4) with quality gating and referral triage."""
    return await _process_aptos_upload(file)

@router.post("/api/predict", response_model=AptosResponse)
async def legacy_predict(image: UploadFile = File(..., description="Legacy field name 'image'")):
    """Legacy endpoint alias for backward compatibility with frontend v1."""
    return await _process_aptos_upload(image)
