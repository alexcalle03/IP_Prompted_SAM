from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "input"
DEFAULT_OUTPUT = PROJECT_ROOT / "output"

from segmenter.segment_anything import (
    _interactive_segmentation,
    list_image_files,
    load_image_array,
    save_mask_image,
    save_mask_overlay,
)


def run_folder_segmenter() -> list[Path]:
    image_paths = list_image_files(DEFAULT_INPUT)
    if not image_paths:
        raise FileNotFoundError(f"No supported image files found in: {DEFAULT_INPUT}")

    target_dir = Path(DEFAULT_OUTPUT) if DEFAULT_OUTPUT is not None else Path(DEFAULT_INPUT)
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for image_path in image_paths:
        if image_path.name.removesuffix(".png") in [f.name.removesuffix("_mask.png") for f in target_dir.iterdir()]:
            print(f"Skipping {image_path.name}: filename already in output folder.")
            continue

        image = load_image_array(image_path)
        points, mask, decision = _interactive_segmentation(image, image_path)

        if decision == "quit":
            print("Quitting segmentation session.")
            break
        if decision == "next":
            print(f"Skipping {image_path.name}.")
            continue
        if decision == "redo":
            print(f"Redoing {image_path.name}.")
            continue
        if len(points) < 3:
            print(f"Skipping {image_path.name}: need at least 3 points.")
            continue

        mask_path = target_dir / f"{image_path.stem}_mask.png"
        overlay_path = target_dir / f"{image_path.stem}_overlay.png"
        save_mask_image(mask, mask_path)
        save_mask_overlay(image, mask, overlay_path, points)
        saved_paths.extend([mask_path, overlay_path])
        print(f"Saved mask to {mask_path}")
        print(f"Saved overlay to {overlay_path}")

    print(f"Segmentation completed. Saved {len(saved_paths)} files to {target_dir}")
    return saved_paths

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive SAM segmentation app")
    parser.add_argument("--folder", type=str, default=str(DEFAULT_INPUT), help="Folder containing images to segment")
    parser.add_argument("--image", type=str, default="", help="Single image to segment")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Folder to save masks")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    print(f"Main project root: {PROJECT_ROOT}")
    print(f"Segmenting images in folder: {DEFAULT_INPUT}")
    run_folder_segmenter()


if __name__ == "__main__":
    main()