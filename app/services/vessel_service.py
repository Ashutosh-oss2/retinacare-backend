import cv2
import numpy as np
import torch
from app.services.model_registry import registry
from app.utils.image_processing import encode_image_to_base64_png
from app.schemas.vessel import VesselSegmentationResponse

CLINICAL_DISCLAIMER = (
    "This system provides AI-assisted retinal screening and segmentation support "
    "and is not a substitute for professional medical diagnosis."
)

def _preprocess_green_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """Green channel extraction + CLAHE adaptive contrast enhancement."""
    green = img_bgr[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    return enhanced.astype(np.float32) / 255.0


def run_vessel_segmentation(img_bgr: np.ndarray, filename: str, threshold: float = 0.5, include_base64: bool = True) -> VesselSegmentationResponse:
    """Executes DRIVE sliding-window U-Net segmentation."""
    if not registry.is_vessel_loaded:
        raise RuntimeError("Model 2 (DRIVE Vessel Segmentation) is not loaded in memory.")

    enhanced = _preprocess_green_clahe(img_bgr)
    h, w = enhanced.shape
    patch_size = 48
    stride = 24
    pad = patch_size
    img_padded = np.pad(enhanced, pad, mode="reflect")

    pred_map = np.zeros_like(img_padded, dtype=np.float32)
    count_map = np.zeros_like(img_padded, dtype=np.float32)

    with torch.no_grad():
        for y in range(0, img_padded.shape[0] - patch_size, stride):
            for x in range(0, img_padded.shape[1] - patch_size, stride):
                patch = img_padded[y : y + patch_size, x : x + patch_size]
                tensor = torch.from_numpy(patch[np.newaxis, np.newaxis, :, :]).float().to(registry.device)
                out = registry.vessel_model(tensor).cpu().numpy()[0, 0]
                pred_map[y : y + patch_size, x : x + patch_size] += out
                count_map[y : y + patch_size, x : x + patch_size] += 1

    count_map[count_map == 0] = 1.0
    pred_map = pred_map / count_map
    vessel_prob = pred_map[pad : pad + h, pad : pad + w]

    # Binary mask uint8
    binary_mask = (vessel_prob >= threshold).astype(np.uint8) * 255

    # Compute retina area (non-black fundus pixels)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    retina_px = int(np.sum(gray > 15))
    if retina_px == 0:
        retina_px = h * w

    vessel_px = int(np.sum(binary_mask > 0))
    vessel_density = round((vessel_px / retina_px) * 100.0, 2)

    base64_mask = encode_image_to_base64_png(binary_mask, is_bgr=True) if include_base64 else None

    return VesselSegmentationResponse(
        success=True,
        filename=filename,
        model="DRIVE Vessel U-Net",
        dimensions=[w, h],
        vessel_pixel_count=vessel_px,
        retina_pixel_count=retina_px,
        vessel_density_percentage=vessel_density,
        mask_base64_png=base64_mask,
        mask_image_url=None,
        disclaimer=CLINICAL_DISCLAIMER
    )
