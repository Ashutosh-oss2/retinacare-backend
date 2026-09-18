"""Analyze focus calibration and quality metrics across real APTOS images.

This script intentionally calls the assessment layer directly. It does not
run the quality pipeline's borderline enhancement step.
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

from .quality_assessment import QualityAssessment, assess_quality
from .utils import load_image

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def image_paths(directory: Path, recursive: bool) -> Iterable[Path]:
    """Yield supported image files from a directory in stable order."""
    paths = directory.rglob("*") if recursive else directory.glob("*")
    return sorted(path for path in paths if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def print_metric_summary(name: str, values: list[float]) -> None:
    """Print the requested descriptive statistics for one metric."""
    numbers = np.asarray(values, dtype=np.float64)
    percentiles = np.percentile(numbers, [10, 25, 50, 75, 90])
    print(f"\n{name}:")
    print(f"  count: {numbers.size}")
    print(f"  minimum: {numbers.min():.6f}")
    print(f"  10th percentile: {percentiles[0]:.6f}")
    print(f"  25th percentile: {percentiles[1]:.6f}")
    print(f"  median: {percentiles[2]:.6f}")
    print(f"  75th percentile: {percentiles[3]:.6f}")
    print(f"  90th percentile: {percentiles[4]:.6f}")
    print(f"  maximum: {numbers.max():.6f}")
    print(f"  mean: {numbers.mean():.6f}")
    print(f"  standard deviation: {numbers.std():.6f}")


def print_raw_focus_summary(values: list[float]) -> None:
    """Print raw focus statistics and counts at the configured focus floor."""
    numbers = np.asarray(values, dtype=np.float64)
    percentiles = np.percentile(numbers, [10, 25, 50, 75, 90])
    print("\nRAW FOCUS MEASURE")
    print("-----------------")
    print(f"count: {numbers.size}")
    print(f"minimum: {numbers.min():.6f}")
    print(f"10th percentile: {percentiles[0]:.6f}")
    print(f"25th percentile: {percentiles[1]:.6f}")
    print(f"median: {percentiles[2]:.6f}")
    print(f"75th percentile: {percentiles[3]:.6f}")
    print(f"90th percentile: {percentiles[4]:.6f}")
    print(f"maximum: {numbers.max():.6f}")
    print(f"mean: {numbers.mean():.6f}")
    print(f"standard deviation: {numbers.std():.6f}")
    print("\nRaw focus measure counts:")
    print(f"  < 10: {np.count_nonzero(numbers < 10)}")
    print(f"  < 20: {np.count_nonzero(numbers < 20)}")
    print(f"  < 35: {np.count_nonzero(numbers < 35)}")
    print(f"  >= 35: {np.count_nonzero(numbers >= 35)}")


def print_image_result(path: Path, assessment: QualityAssessment) -> None:
    """Print the collected assessment for one image."""
    print(
        f"{path}: "
        f"focus_measure={assessment.focus_measure:.6f}, "
        f"focus_score={assessment.focus_score:.6f}, "
        f"illumination_score={assessment.illumination_score:.6f}, "
        f"field_of_view_score={assessment.field_of_view_score:.6f}, "
        f"quality_score={assessment.quality_score:.6f}, "
        f"status={assessment.quality_status.value}"
    )


def analyze_directory(directory: Path, recursive: bool = False) -> int:
    """Assess every supported image and print metric distributions."""
    paths = list(image_paths(directory, recursive))
    if not paths:
        print(f"No supported images found in: {directory}", file=sys.stderr)
        return 1

    focus_measures: list[float] = []
    focus_scores: list[float] = []
    illumination_scores: list[float] = []
    field_of_view_scores: list[float] = []
    quality_scores: list[float] = []
    processed_count = 0

    print(f"Analyzing {len(paths)} image(s) in: {directory}")
    print("Assessment only; no enhancement is performed.\n")
    for path in paths:
        try:
            assessment = assess_quality(load_image(path))
        except (OSError, ValueError) as error:
            print(f"Skipping {path}: {error}", file=sys.stderr)
            continue

        print_image_result(path, assessment)
        focus_measures.append(assessment.focus_measure)
        focus_scores.append(assessment.focus_score)
        illumination_scores.append(assessment.illumination_score)
        field_of_view_scores.append(assessment.field_of_view_score)
        quality_scores.append(assessment.quality_score)
        processed_count += 1

    if not processed_count:
        print("No images could be assessed.", file=sys.stderr)
        return 1

    print(f"\nSuccessfully assessed: {processed_count}/{len(paths)}")
    print_raw_focus_summary(focus_measures)
    print_metric_summary("Illumination score", illumination_scores)
    print_metric_summary("Field-of-view score", field_of_view_scores)
    print_metric_summary("Focus score", focus_scores)
    print_metric_summary("Overall quality score", quality_scores)
    return 0


def main() -> int:
    """Parse the image directory and run the analysis."""
    parser = argparse.ArgumentParser(
        description="Summarize existing image-quality metrics for APTOS images."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing APTOS image files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include supported images in subdirectories.",
    )
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error(f"Not a directory: {args.directory}")
    return analyze_directory(args.directory, args.recursive)


if __name__ == "__main__":
    raise SystemExit(main())
