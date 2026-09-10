from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.utils import copy_sample, infer_split_dir, reset_directory


SEED = 42
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def split_mapping() -> dict[str, tuple[str, ...]]:
    return {
        "train": ("train", "Train"),
        "valid": ("valid", "val", "Valid", "Val"),
        "test": ("test", "Test"),
    }


def polygon_to_mask(height: int, width: int, segmentations: list[list[float]]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for segmentation in segmentations:
        if not segmentation:
            continue
        polygon = np.array(segmentation, dtype=np.float32).reshape(-1, 2).astype(np.int32)
        if polygon.size:
            cv2.fillPoly(mask, [polygon], 255)
    return mask


def collect_roboflow_samples(raw_root: Path) -> list[dict]:
    samples: list[dict] = []
    roboflow_root = raw_root / "roboflow"

    for split_candidates in split_mapping().values():
        split_dir = infer_split_dir(roboflow_root, split_candidates)
        if split_dir is None:
            continue

        annotation_path = split_dir / "_annotations.coco.json"
        if not annotation_path.exists():
            continue

        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        images = {item["id"]: item for item in data.get("images", [])}
        annotations_by_image: dict[int, list] = defaultdict(list)

        for annotation in data.get("annotations", []):
            if annotation.get("segmentation"):
                annotations_by_image[annotation["image_id"]].extend(annotation["segmentation"])

        for image_id, segmentations in annotations_by_image.items():
            image_info = images.get(image_id)
            if image_info is None:
                continue

            image_path = split_dir / image_info["file_name"]
            if not image_path.exists():
                continue

            mask = polygon_to_mask(int(image_info["height"]), int(image_info["width"]), segmentations)
            if not np.any(mask):
                continue

            samples.append(
                {
                    "source": "roboflow",
                    "image_path": image_path,
                    "image_name": f"roboflow_{image_path.stem}{image_path.suffix.lower()}",
                    "mask_name": f"roboflow_{image_path.stem}.png",
                    "mask_array": mask,
                }
            )

    return samples


def collect_lados_samples(raw_root: Path) -> list[dict]:
    samples: list[dict] = []
    lados_root = raw_root / "lados"

    for split_candidates in split_mapping().values():
        split_dir = infer_split_dir(lados_root, split_candidates)
        if split_dir is None:
            continue

        for image_path in sorted(split_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
                continue
            if image_path.stem.endswith("_mask"):
                continue

            mask_path = image_path.with_name(f"{image_path.stem}_mask.png")
            if not mask_path.exists() or mask_path.stat().st_size == 0:
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                continue

            image_height, image_width = image.shape[:2]
            if mask.shape[:2] != (image_height, image_width):
                mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)

            mask = np.where(mask > 0, 255, 0).astype(np.uint8)
            samples.append(
                {
                    "source": "lados",
                    "image_path": image_path,
                    "image_name": f"lados_{image_path.stem}{image_path.suffix.lower()}",
                    "mask_name": f"lados_{image_path.stem}.png",
                    "mask_array": mask,
                }
            )

    return samples


def stratified_split(samples: list[dict]) -> dict[str, list[dict]]:
    if len(samples) < 3:
        raise ValueError("Need at least 3 samples to create train/val/test splits.")

    labels = [sample["source"] for sample in samples]
    train_samples, temp_samples = train_test_split(
        samples,
        test_size=0.30,
        random_state=SEED,
        shuffle=True,
        stratify=labels,
    )
    temp_labels = [sample["source"] for sample in temp_samples]
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=0.50,
        random_state=SEED,
        shuffle=True,
        stratify=temp_labels,
    )

    return {"train": train_samples, "val": val_samples, "test": test_samples}


def materialize_splits(samples_by_split: dict[str, list[dict]], output_root: Path) -> None:
    reset_directory(output_root)
    for split_name in ("train", "val", "test"):
        (output_root / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split_name / "masks").mkdir(parents=True, exist_ok=True)

    for split_name, records in samples_by_split.items():
        for record in records:
            image_output = output_root / split_name / "images" / record["image_name"]
            mask_output = output_root / split_name / "masks" / record["mask_name"]
            copy_sample(record["image_path"], image_output)
            Image.fromarray(record["mask_array"], mode="L").save(mask_output)


def mask_stats(records: list[dict]) -> dict[str, float | int]:
    if not records:
        return {"count": 0, "avg_foreground_pct": 0.0, "empty_masks": 0, "full_masks": 0}

    coverages = []
    empty_masks = 0
    full_masks = 0
    for record in records:
        mask = record["mask_array"]
        foreground_pct = float((mask > 0).sum() / mask.size * 100.0)
        coverages.append(foreground_pct)
        if foreground_pct == 0.0:
            empty_masks += 1
        if foreground_pct == 100.0:
            full_masks += 1

    return {
        "count": len(records),
        "avg_foreground_pct": round(float(np.mean(coverages)), 4),
        "empty_masks": empty_masks,
        "full_masks": full_masks,
    }


def summarize(samples: list[dict], samples_by_split: dict[str, list[dict]]) -> dict:
    return {
        "task": "segmentation",
        "sources": {
            "roboflow": sum(1 for sample in samples if sample["source"] == "roboflow"),
            "lados": sum(1 for sample in samples if sample["source"] == "lados"),
        },
        "total_samples": len(samples),
        "splits": {
            split_name: {
                "samples": len(records),
                "source_breakdown": {
                    "roboflow": sum(1 for sample in records if sample["source"] == "roboflow"),
                    "lados": sum(1 for sample in records if sample["source"] == "lados"),
                },
                "mask_stats": mask_stats(records),
            }
            for split_name, records in samples_by_split.items()
        },
        "global_mask_stats": mask_stats(samples),
    }


def main() -> None:
    project_root = PROJECT_ROOT
    raw_root = project_root / "data" / "raw"
    output_root = project_root / "data" / "splits" / "segmentation"
    report_root = project_root / "outputs" / "pipeline_build"
    report_root.mkdir(parents=True, exist_ok=True)

    samples = collect_roboflow_samples(raw_root) + collect_lados_samples(raw_root)
    random.Random(SEED).shuffle(samples)
    samples_by_split = stratified_split(samples)
    materialize_splits(samples_by_split, output_root)

    summary = summarize(samples, samples_by_split)
    (report_root / "segmentation_build_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
