import cv2
import numpy as np
import torch
from app.services.model_registry import registry
from app.utils.image_processing import encode_image_to_base64_png
from app.schemas.idrid import IdridSegmentationResponse, LesionItem

CLINICAL_DISCLAIMER = (
    "This system provides AI-assisted retinal screening and segmentation support "
    "and is not a substitute for professional medical diagnosis."
)

LESION_TYPES = ["MA", "HE", "EX", "SE"]
LESION_NAMES_FULL = {
    "MA": "Microaneurysms (MA)",
    "HE": "Hemorrhages (HE)",
    "EX": "Hard Exudates (EX)",
    "SE": "Soft Exudates (SE)"
}

LESION_COLORS_RGB = {
    "MA": (255, 0, 0),      # Microaneurysms: Bright Red
    "HE": (200, 0, 100),    # Hemorrhages: Deep Red / Magenta
    "EX": (255, 255, 0),    # Hard Exudates: Yellow
    "SE": (0, 255, 255)     # Soft Exudates: Cyan
}

def _apply_clahe_lab(img_bgr: np.ndarray) -> np.ndarray:
    """Enhances L channel in LAB color space."""
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)


def _create_color_overlay(img_rgb: np.ndarray, binary_masks: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Creates composite multi-lesion color-coded overlay."""
    overlay = img_rgb.copy().astype(np.float32)
    color_mask = np.zeros_like(img_rgb, dtype=np.float32)
    has_lesion = np.zeros(img_rgb.shape[:2], dtype=bool)

    for i, code in enumerate(LESION_TYPES):
        mask_i = binary_masks[i] > 0
        if np.any(mask_i):
            color = LESION_COLORS_RGB[code]
            color_mask[mask_i] = color
            has_lesion[mask_i] = True

    overlay[has_lesion] = (1.0 - alpha) * overlay[has_lesion] + alpha * color_mask[has_lesion]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def run_idrid_segmentation(img_bgr: np.ndarray, filename: str, threshold: float = 0.5, include_base64: bool = True) -> IdridSegmentationResponse:
    """Executes IDRiD 4-channel lesion segmentation."""
    if not registry.is_idrid_loaded:
        raise RuntimeError("Model 3 (IDRiD Lesion Segmentation) is not loaded in memory.")

    orig_h, orig_w = img_bgr.shape[:2]
    enhanced_rgb = _apply_clahe_lab(img_bgr)

    # Resize for UNet
    img_size = (256, 256)
    resized_rgb = cv2.resize(enhanced_rgb, img_size, interpolation=cv2.INTER_LINEAR)

    # Tensor normalization (ImageNet standards)
    img_tensor = torch.from_numpy(resized_rgb.transpose(2, 0, 1)).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    img_tensor = img_tensor.unsqueeze(0).to(registry.device)

    with torch.no_grad():
        logits = registry.idrid_model(img_tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    binary_masks_resized = (probs >= threshold).astype(np.uint8)
    binary_masks_orig = np.zeros((4, orig_h, orig_w), dtype=np.uint8)

    # Estimate fundus retina area (non-black pixels)
    gray_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    retina_pixel_count = int(np.sum(gray_orig > 15))
    if retina_pixel_count == 0:
        retina_pixel_count = orig_h * orig_w

    lesions_dict = {}
    total_lesion_pixels_union = np.zeros((orig_h, orig_w), dtype=bool)

    for i, code in enumerate(LESION_TYPES):
        m_orig = cv2.resize(binary_masks_resized[i], (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        binary_masks_orig[i] = m_orig
        px_count = int(np.sum(m_orig > 0))
        area_pct = round((px_count / retina_pixel_count) * 100.0, 4)

        if px_count > 0:
            total_lesion_pixels_union = total_lesion_pixels_union | (m_orig > 0)

        lesions_dict[code] = LesionItem(
            name=LESION_NAMES_FULL[code],
            detected=bool(px_count > 0),
            pixel_count=px_count,
            area_percentage=area_pct
        )

    total_lesion_area = round((int(np.sum(total_lesion_pixels_union)) / retina_pixel_count) * 100.0, 2)

    # Clinical risk stratification
    he_detected = lesions_dict["HE"].detected
    se_detected = lesions_dict["SE"].detected
    ex_detected = lesions_dict["EX"].detected

    if total_lesion_area > 5.0 or (he_detected and se_detected):
        clinical_risk_level = "High Lesion Burden / Severe Risk"
        clinical_action = "Immediate ophthalmologist referral recommended for full comprehensive evaluation."
    elif total_lesion_area > 1.0 or (he_detected or ex_detected):
        clinical_risk_level = "Moderate Lesion Burden / Moderate Risk"
        clinical_action = "Ophthalmologist referral advised for clinical staging."
    elif total_lesion_area > 0.0:
        clinical_risk_level = "Mild Lesion Signs Detected"
        clinical_action = "Routine monitoring and diabetic glycemic control advised."
    else:
        clinical_risk_level = "No Significant Retinal Lesions Detected"
        clinical_action = "Annual routine diabetic retinopathy screening recommended."

    orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    overlay_rgb = _create_color_overlay(orig_rgb, binary_masks_orig, alpha=0.55)
    base64_overlay = encode_image_to_base64_png(overlay_rgb, is_bgr=False) if include_base64 else None

    return IdridSegmentationResponse(
        success=True,
        filename=filename,
        dimensions=[orig_w, orig_h],
        lesions=lesions_dict,
        total_lesion_area=total_lesion_area,
        clinical_risk_level=clinical_risk_level,
        clinical_action=clinical_action,
        overlay_base64_png=base64_overlay,
        overlay_image_url=None,
        disclaimer=CLINICAL_DISCLAIMER
    )
