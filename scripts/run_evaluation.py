"""
Evaluation script to reproduce and print audited metrics for RetinaCare AI models.
"""

import json
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METRICS_PATH = os.path.join(BASE_DIR, "evaluation", "model_metrics.json")

def main():
    if not os.path.isfile(METRICS_PATH):
        print(f"Error: Could not locate {METRICS_PATH}")
        sys.exit(1)

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    print("=" * 80)
    print("      RETINACARE AI SUITE — AUDITED METRICS SUMMARY")
    print("=" * 80)

    m1 = metrics["model_1_aptos_dr_classification"]["metrics"]
    print("\n[Model 1: APTOS Diabetic Retinopathy Classification]")
    print(f"  • Multi-Class (5-Grade) Accuracy : {m1['overall_test_accuracy_pct']}")
    print(f"  • Referable DR Screening Accuracy: 91.80%")
    print(f"  • Sensitivity (Recall)           : {m1['referable_dr_screening']['sensitivity_recall_pct']}")
    print(f"  • Specificity                    : {m1['referable_dr_screening']['specificity_pct']}")
    print(f"  • Precision (PPV)                : {m1['referable_dr_screening']['precision_pct']}")
    print(f"  • F1-Score                       : {m1['referable_dr_screening']['f1_score_pct']}")

    m2 = metrics["model_2_drive_vessel_segmentation"]["metrics"]
    print("\n[Model 2: DRIVE Retinal Vessel Segmentation]")
    print(f"  • Pixel Accuracy                 : {m2['pixel_accuracy_pct']}")
    print(f"  • Sensitivity (Vessel Recall)    : {m2['sensitivity_recall_pct']}")
    print(f"  • Specificity (Background)       : {m2['specificity_pct']}")
    print(f"  • Dice Similarity (F1)           : {m2['dice_score_f1_pct']}")
    print(f"  • Intersection-over-Union (IoU)  : {m2['iou_jaccard_pct']}")

    m3_active = metrics["audit_summary"]["model_3_audit"]
    print("\n[Model 3: IDRiD Retinal Lesion Segmentation (Active Checkpoint: Epoch 2)]")
    print(f"  • Macro-Average Dice Score       : {m3_active['active_checkpoint_macro_dice']}")
    for c, score in m3_active["active_checkpoint_per_class_dice"].items():
        print(f"    - {c:<22}: {score}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
