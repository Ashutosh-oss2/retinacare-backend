"""Run the complete quality pipeline over a directory of fundus images."""

import argparse
import sys
from pathlib import Path
from typing import Iterable

from .pipeline import assess_image_quality, process_image
from .quality_assessment import QualityStatus

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def image_paths(directory: Path, recursive: bool) -> Iterable[Path]:
    """Yield supported image files from a directory in stable order."""
    paths = directory.rglob("*") if recursive else directory.glob("*")
    return sorted(
        path for path in paths
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def analyze_directory(directory: Path, recursive: bool = False) -> int:
    """Run initial assessment and the complete pipeline for each image."""
    paths = list(image_paths(directory, recursive))
    if not paths:
        print(f"No supported images found in: {directory}", file=sys.stderr)
        return 1

    counts = {
        QualityStatus.ACCEPTABLE.value: 0,
        QualityStatus.BORDERLINE.value: 0,
        QualityStatus.UNGRADABLE.value: 0,
        "enhanced": 0,
        "recapture_required": 0,
    }
    assessed_count = 0

    print(f"Processing {len(paths)} image(s) in: {directory}")
    print("Complete quality pipeline: assessment, enhancement, reassessment.\n")

    for path in paths:
        try:
            original = assess_image_quality(path)
            result = process_image(path)
            data = result.to_dict()
        except (OSError, ValueError) as error:
            print(f"Skipping {path.name}: {error}", file=sys.stderr)
            continue

        final_status = data["quality_status"]
        enhancements = data.get("enhancements_used", [])
        recapture_required = bool(data.get("recapture_required", False))
        enhancement_applied = bool(data.get("enhancement_applied", False))

        print(f"filename: {path.name}")
        print(f"original_quality_score: {original.quality_score:.6f}")
        print(f"original_focus_score: {original.focus_score:.6f}")
        print(f"final_quality_score: {data['quality_score']:.6f}")
        print(f"final_focus_score: {data['focus_score']:.6f}")
        print(f"enhancement_applied: {enhancement_applied}")
        print(f"enhancements: {', '.join(enhancements) if enhancements else 'none'}")
        print(f"final_status: {final_status}")
        print(f"recapture_required: {recapture_required}\n")

        counts[final_status] += 1
        counts["enhanced"] += int(enhancement_applied)
        counts["recapture_required"] += int(recapture_required)
        assessed_count += 1

    if not assessed_count:
        print("No images could be assessed.", file=sys.stderr)
        return 1

    print("COUNTS")
    print("------")
    print(f"ACCEPTABLE: {counts[QualityStatus.ACCEPTABLE.value]}")
    print(f"BORDERLINE: {counts[QualityStatus.BORDERLINE.value]}")
    print(f"UNGRADABLE: {counts[QualityStatus.UNGRADABLE.value]}")
    print(f"ENHANCED: {counts['enhanced']}")
    print(f"RECAPTURE_REQUIRED: {counts['recapture_required']}")
    return 0


def main() -> int:
    """Parse command-line arguments and run the batch pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the complete image-quality pipeline on fundus images."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing fundus image files.",
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
