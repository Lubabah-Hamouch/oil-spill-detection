from __future__ import annotations

import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_SIZE = 224
RANDOM_SEED = 42
CLASSIFICATION_SPLITS = ("train", "val", "test")
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def list_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(
        file_path
        for file_path in path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )


def copy_sample(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def infer_split_dir(root: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        candidate = root / name
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def image_has_foreground(mask_path: Path) -> bool:
    mask = np.array(Image.open(mask_path).convert("L"))
    return bool(mask.max() > 0)



class ClassificationSplitDataset(Dataset):
    def __init__(self, root: Path, split: str, train: bool) -> None:
        self.root = root / split
        self.samples: list[tuple[Path, int]] = []
        for class_id in (0, 1):
            for image_path in list_images(self.root / str(class_id)):
                self.samples.append((image_path, class_id))

        if train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label, str(path)

    def class_counts(self) -> dict[int, int]:
        counts = Counter(label for _, label in self.samples)
        return {0: counts.get(0, 0), 1: counts.get(1, 0)}

    def class_weights(self) -> torch.Tensor:
        counts = self.class_counts()
        total = max(sum(counts.values()), 1)
        weights = []
        for class_id in (0, 1):
            count = counts[class_id]
            weights.append(total / (2.0 * count) if count else 1.0)
        return torch.tensor(weights, dtype=torch.float32)


class SegmentationSplitDataset(Dataset):
    def __init__(self, root: Path, split: str, train: bool) -> None:
        self.image_dir = root / split / "images"
        self.mask_dir = root / split / "masks"
        self.samples: list[tuple[Path, Path]] = []

        for image_path in list_images(self.image_dir):
            mask_path = self.mask_dir / f"{image_path.stem}.png"
            if mask_path.exists():
                self.samples.append((image_path, mask_path))

        self.resize_image = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
        self.resize_mask = transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=transforms.InterpolationMode.NEAREST,
        )
        self.train = train
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.train and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.train and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        image = self.resize_image(image)
        mask = self.resize_mask(mask)
        image_tensor = self.to_tensor(image)
        mask_tensor = torch.from_numpy((np.array(mask) > 0).astype(np.float32)).unsqueeze(0)
        return image_tensor, mask_tensor, str(image_path), str(mask_path)
