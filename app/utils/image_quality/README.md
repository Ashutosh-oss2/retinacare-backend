# Image Quality Module

A standalone OpenCV/NumPy module for fundus image quality control. It does not import or perform APTOS preprocessing and has no dependency on the existing backend or model.

## Flow

`image -> focus + illumination + field of view -> quality decision -> optional borderline enhancement`

The module returns `acceptable`, `borderline`, or `ungradable`:

- **Acceptable** images pass through unchanged.
- **Borderline** images receive only indicated CLAHE, illumination normalization, and/or denoising. The result can be passed to a model-specific preprocessing pipeline later.
- **Ungradable** images are rejected without repeated enhancement and include recapture feedback.

Black background pixels are excluded from retinal illumination and focus statistics. Field-of-view scoring uses the largest plausible non-black connected region and checks its area, bounding box, and crop against the image edges.

## Usage

```python
from image_quality import process_image

result = process_image("fundus.jpg", output_path="quality_controlled.jpg")
print(result.assessment.to_dict())
print(result.enhancements_used)
if result.recapture_required:
    print(result.recapture_feedback)
```

`result.processed_image` is a BGR `numpy.ndarray`; `result.processed_image_path` is set when `output_path` is supplied. Thresholds and enhancement parameters are configurable through `QualityConfig`.

## Tests

From the repository root:

```bash
python -m pytest image_quality/test_quality.py -q
```

The test suite covers good, blurry, dark, overexposed, unevenly illuminated, insufficient-field-of-view, borderline, and rejected images.
