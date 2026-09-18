import io
import os
import base64
import numpy as np
import cv2
from fastapi import UploadFile, HTTPException
from app.utils.logging import logger

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

async def validate_and_decode_image(file: UploadFile) -> tuple[np.ndarray, bytes, str]:
    """
    Validates uploaded image file size, filename, and extension,
    then decodes the byte buffer to a BGR numpy image array.
    Returns (image_bgr, contents_bytes, filename).
    """
    if not file.filename:
        logger.warning("Upload rejected: missing filename")
        raise HTTPException(
            status_code=400,
            detail={"error": "missing_filename", "message": "No image filename provided."}
        )

    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        logger.warning(f"Upload rejected: unsupported extension '{extension}'")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_format",
                "message": f"Unsupported format '{extension}'. Accepted formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            }
        )

    contents = await file.read()
    if not contents or len(contents) == 0:
        logger.warning("Upload rejected: empty file payload")
        raise HTTPException(
            status_code=400,
            detail={"error": "empty_file", "message": "The uploaded file is empty."}
        )

    if len(contents) > MAX_FILE_SIZE:
        size_mb = round(len(contents) / (1024 * 1024), 2)
        logger.warning(f"Upload rejected: file size {size_mb}MB exceeds limit")
        raise HTTPException(
            status_code=413,
            detail={"error": "file_too_large", "message": f"Image size {size_mb} MB exceeds the 10 MB limit."}
        )

    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None or image_bgr.size == 0:
        logger.warning("Upload rejected: corrupt or undecodable image buffer")
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_image", "message": "Corrupt or undecodable image data."}
        )

    return image_bgr, contents, file.filename


def encode_image_to_base64_png(image_array: np.ndarray, is_bgr: bool = True) -> str:
    """Encodes a numpy image (BGR or RGB) into a base64 PNG data URL string."""
    if is_bgr:
        img_to_encode = image_array
    else:
        img_to_encode = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    success, buffer = cv2.imencode(".png", img_to_encode)
    if not success:
        raise ValueError("Failed to encode image to PNG buffer.")

    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"
