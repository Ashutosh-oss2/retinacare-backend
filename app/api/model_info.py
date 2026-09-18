from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter(tags=["Model Metadata & Benchmarks"])

CLINICAL_DISCLAIMER = (
    "This system provides AI-assisted retinal screening and segmentation support "
    "and is not a substitute for professional medical diagnosis."
)

AUDITED_SUITE_METRICS: Dict[str, Any] = {
    "disclaimer": CLINICAL_DISCLAIMER,
    "model_1_aptos": {
        "model": "EfficientNetB0",
        "task": "Diabetic Retinopathy Classification",
        "dataset": "APTOS 2019 Blindness Detection",
        "input_size": "224x224",
        "classes": {
            "0": "No DR",
            "1": "Mild DR",
            "2": "Moderate DR",
            "3": "Severe DR",
            "4": "Proliferative DR"
        },
        "referable_definition": "DR Grade 2 or higher (Moderate, Severe, Proliferative)",
        "referable_threshold": 0.42,
        "test_dataset_size": 366,
        "referable_screening_accuracy": 0.9180,
        "sensitivity": 0.9343,
        "specificity": 0.9083,
        "precision": 0.8591,
        "f1_score": 0.8951,
        "multiclass_accuracy": 0.7623,
    },
    "model_2_drive": {
        "model": "3-Level PyTorch U-Net",
        "task": "Retinal Vessel Segmentation",
        "dataset": "Digital Retinal Images for Vessel Extraction (DRIVE)",
        "pixel_accuracy": 0.9535,
        "sensitivity": 0.7765,
        "specificity": 0.9771,
        "precision": 0.8191,
        "dice_score": 0.7972,
        "iou_jaccard": 0.6628,
        "roc_auc": 0.9710,
    },
    "model_3_idrid": {
        "model": "4-Channel PyTorch U-Net",
        "task": "Multi-Lesion Retinal Segmentation",
        "dataset": "Indian Diabetic Retinopathy Image Dataset (IDRiD)",
        "active_checkpoint": "idrid_best_model.pth (Epoch 2)",
        "macro_dice": 0.3435,
        "macro_pixel_accuracy": 0.9886,
        "per_class_dice": {
            "HE_hemorrhages": 0.7421,
            "SE_soft_exudates": 0.5182,
            "EX_hard_exudates": 0.1138,
            "MA_microaneurysms": 0.0000,
        },
        "known_limitations": "Microaneurysms (0.12% pixel area) require high-resolution patch magnification (>1024x1024).",
    }
}

@router.get("/api/model-info")
def get_model_info():
    return {
        "models": ["APTOS EfficientNetB0", "DRIVE 3-Level U-Net", "IDRiD 4-Channel U-Net"],
        "disclaimer": CLINICAL_DISCLAIMER,
        "details": AUDITED_SUITE_METRICS,
    }

@router.get("/api/model-stats")
def get_model_stats():
    return AUDITED_SUITE_METRICS
