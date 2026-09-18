from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from image_quality.pipeline import process_image


# ============================================================
# REAL APTOS IMAGE
# ============================================================

IMAGE_PATH = Path(
    r"C:\Users\DELL\Downloads\001639a390f0.png"
)


# ============================================================
# PROCESS IMAGE
# ============================================================

print("=" * 60)
print("IMAGE QUALITY TEST")
print("=" * 60)

print("Image:", IMAGE_PATH)

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )


result = process_image(
    str(IMAGE_PATH)
)
data = result.to_dict()


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 60)
print("QUALITY RESULT")
print("=" * 60)

print(
    "Quality status:",
    data.get("quality_status")
)

print(
    "Overall quality score:",
    data.get("quality_score")
)

print(
    "Focus score:",
    data.get("focus_score")
)

print(
    "Focus status:",
    "acceptable" if data.get("focus_score", 0.0) >= 0.52 else "needs attention"
)

print(
    "Illumination score:",
    data.get("illumination_score")
)

print(
    "Illumination status:",
    "needs attention" if (
        data.get("underexposure") or
        data.get("overexposure") or
        data.get("uneven_illumination")
    ) else "acceptable"
)

print(
    "Field-of-view score:",
    data.get("field_of_view_score")
)

print(
    "Field-of-view status:",
    "needs attention" if (
        data.get("insufficient_field_of_view") or
        data.get("severely_cropped")
    ) else "acceptable"
)

print(
    "Enhancement applied:",
    data.get(
        "enhancement_applied",
        False
    )
)

print(
    "Enhancements:",
    data.get(
        "enhancements_used",
        []
    )
)

print(
    "Recapture required:",
    data.get(
        "recapture_required",
        False
    )
)

print(
    "Recapture feedback:",
    data.get(
        "recapture_feedback"
    )
)
print(
    "Raw focus measure:",
    data.get("focus_measure")
)