import cv2
import numpy as np
from app.utils.logging import logger
from app.services.model_registry import registry
from app.utils.image_quality.pipeline import process_image
from app.schemas.aptos import AptosResponse, AptosPrediction, ImageQualityInfo

CLASS_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

CLASS_DESCRIPTIONS = {
    0: "No signs of diabetic retinopathy detected.",
    1: "Mild diabetic retinopathy — microaneurysms only.",
    2: "Moderate diabetic retinopathy — more than mild, less than severe.",
    3: "Severe diabetic retinopathy — extensive hemorrhages; high risk of progression.",
    4: "Proliferative diabetic retinopathy — most advanced stage; neovascularization present.",
}

REFERABLE_THRESHOLD = 0.42
CLINICAL_DISCLAIMER = (
    "This system provides AI-assisted retinal screening and segmentation support "
    "and is not a substitute for professional medical diagnosis."
)

def _preprocess_fundus_bgr(img_bgr: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Cropping non-black retinal region and aspect-ratio padding."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)

    if coords is None:
        cropped = img_bgr
    else:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = img_bgr[y : y + h, x : x + w]

    h_c, w_c = cropped.shape[:2]
    scale = min(target_size / w_c, target_size / h_c)
    new_w = int(w_c * scale)
    new_h = int(h_c * scale)
    resized = cv2.resize(cropped, (new_w, new_h))

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
    return canvas_rgb


def run_aptos_inference(img_bgr: np.ndarray, filename: str) -> AptosResponse:
    """Executes quality assessment and EfficientNetB0 classification."""
    if not registry.is_aptos_loaded:
        raise RuntimeError("Model 1 (APTOS DR Classifier) is not loaded in memory.")

    # 1. Quality Assessment
    quality_result = process_image(img_bgr)
    quality_dict = quality_result.to_dict()

    quality_info = ImageQualityInfo(
        status=quality_dict["quality_status"],
        quality_score=round(float(quality_dict["quality_score"]), 4),
        focus_score=round(float(quality_dict["focus_score"]), 4),
        illumination_score=round(float(quality_dict["illumination_score"]), 4),
        field_of_view_score=round(float(quality_dict["field_of_view_score"]), 4),
        enhancement_applied=bool(quality_dict["enhancement_applied"]),
        enhancements=list(quality_dict["enhancements_used"]),
        recapture_required=bool(quality_dict["recapture_required"]),
        recapture_feedback=str(quality_dict["recapture_feedback"]),
    )

    if quality_info.recapture_required:
        return AptosResponse(
            success=True,
            filename=filename,
            image_quality=quality_info,
            prediction=None,
            disclaimer=CLINICAL_DISCLAIMER
        )

    # Use quality-controlled enhanced image if available
    img_for_inference = quality_result.processed_image if quality_result.processed_image is not None else img_bgr

    # 2. Preprocessing & Prediction
    preprocessed = _preprocess_fundus_bgr(img_for_inference, target_size=224)
    batch_input = np.expand_dims(preprocessed, axis=0)

    probabilities = registry.aptos_model.predict(batch_input, verbose=0)[0]
    predicted_class = int(np.argmax(probabilities))
    predicted_label = CLASS_NAMES[predicted_class]
    class_description = CLASS_DESCRIPTIONS[predicted_class]
    confidence = float(probabilities[predicted_class])

    referable_probability = float(np.sum(probabilities[2:5]))
    is_referable = referable_probability >= REFERABLE_THRESHOLD
    referable_label = "Referable DR" if is_referable else "Non-Referable DR"

    class_probabilities = {
        CLASS_NAMES[i]: round(float(prob), 4)
        for i, prob in enumerate(probabilities)
    }

    prediction = AptosPrediction(
        predicted_class=predicted_class,
        predicted_label=predicted_label,
        class_description=class_description,
        confidence=round(confidence * 100, 2),
        class_probabilities=class_probabilities,
        referable_probability=round(referable_probability * 100, 2),
        referable_threshold=round(REFERABLE_THRESHOLD * 100, 2),
        referable=is_referable,
        referable_label=referable_label,
        disclaimer=CLINICAL_DISCLAIMER
    )

    return AptosResponse(
        success=True,
        filename=filename,
        image_quality=quality_info,
        prediction=prediction,
        disclaimer=CLINICAL_DISCLAIMER
    )
