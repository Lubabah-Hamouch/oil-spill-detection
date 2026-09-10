"""
Build classification dataset from segmentation data using patch extraction.
Ensures NO DATA LEAKAGE by following segmentation splits (image-level splitting).
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration
PATCH_SIZE = 224
PATCHES_PER_IMAGE = 10  # average patches to attempt per image
OIL_THRESHOLD = 0.05   # 5% threshold for positive class
RANDOM_SEED = 42

class PatchExtractor:
    def __init__(self, image_path, mask_path, patch_size=224, patches_per_image=10):
        self.image = cv2.imread(image_path)
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        
        if self.image is None or self.mask is None:
            raise ValueError(f"Failed to load image or mask: {image_path}")
        
        # Convert image to RGB
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Ensure image and mask have same size
        if self.image.shape[:2] != self.mask.shape:
            self.mask = cv2.resize(self.mask, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    def get_random_patches(self):
        """Extract random patches from image and mask."""
        h, w = self.image.shape[:2]
        patches = []
        max_attempts = self.patches_per_image * 4
        attempts = 0
        
        while len(patches) < self.patches_per_image and attempts < max_attempts:
            if w <= self.patch_size or h <= self.patch_size:
                # Image too small for patch, just take one resize if possible
                patch_img = cv2.resize(self.image, (self.patch_size, self.patch_size))
                patch_mask = cv2.resize(self.mask, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
                patches.append((patch_img, patch_mask))
                break

            x = np.random.randint(0, w - self.patch_size)
            y = np.random.randint(0, h - self.patch_size)
            
            patch_img = self.image[y:y+self.patch_size, x:x+self.patch_size]
            patch_mask = self.mask[y:y+self.patch_size, x:x+self.patch_size]
            
            patches.append((patch_img, patch_mask))
            attempts += 1
        
        return patches

    def get_hard_negative_patches(self, num_patches=4):
        """Extract hard negative patches (high texture or boundary regions)."""
        h, w = self.image.shape[:2]
        if w <= self.patch_size or h <= self.patch_size:
            return []

        hard_patches = []
        
        # 1. Boundary Negatives (near oil but no oil)
        kernel = np.ones((50, 50), np.uint8)
        dilated_mask = cv2.dilate(self.mask, kernel, iterations=1)
        boundary_region = cv2.bitwise_and(dilated_mask, cv2.bitwise_not(self.mask))
        
        # 2. High Texture Negatives (Wave foam, coastlines via Edge detection)
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 30, 100) # Lower thresholds to capture more texture
        
        combined_score = (boundary_region.astype(float) / 255.0) + (edges.astype(float) / 255.0)
        
        coords = np.argwhere(combined_score > 0.5)
        if len(coords) > 0:
            # Shuffle and sample
            indices = np.random.choice(len(coords), min(len(coords), num_patches * 5), replace=False)
            for idx in indices:
                y, x = coords[idx]
                y_start = max(0, min(y - self.patch_size // 2, h - self.patch_size))
                x_start = max(0, min(x - self.patch_size // 2, w - self.patch_size))
                
                patch_mask = self.mask[y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]
                
                if np.count_nonzero(patch_mask) == 0:
                    patch_img = self.image[y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]
                    hard_patches.append((patch_img, patch_mask))
                    if len(hard_patches) >= num_patches:
                        break
        
        return hard_patches

def compute_oil_percentage(mask_patch):
    """Compute percentage of oil pixels (non-zero) in mask patch."""
    total_pixels = mask_patch.size
    oil_pixels = np.count_nonzero(mask_patch)
    return oil_pixels / total_pixels if total_pixels > 0 else 0.0

def build_classification_dataset(data_root):
    """Build classification dataset with zero leakage."""
    print("[START] INITIALIZING LEAKAGE-FREE CLASSIFICATION PIPELINE")
    
    data_root = Path(data_root)
    segmentation_root = data_root / "splits" / "segmentation"
    classification_root = data_root / "splits" / "classification"
    
    # Track unique images per split for verification
    split_source_images = {split: set() for split in ['train', 'val', 'test']}
    
    for split in ['train', 'val', 'test']:
        print(f"\n[SPLIT] PROCESSING {split.upper()} SPLIT...")
        
        split_images_dir = segmentation_root / split / "images"
        split_masks_dir = segmentation_root / split / "masks"
        
        if not split_images_dir.exists():
            print(f"[WARN] {split} split not found in segmentation data, skipping...")
            continue
            
        # Collect patches for this split
        split_patches = {0: [], 1: []}
        
        image_files = sorted(list(split_images_dir.glob("*.jpg")) + list(split_images_dir.glob("*.png")))
        
        ignored_count = 0
        hard_neg_count = 0
        for img_path in tqdm(image_files, desc=f"Extracting patches from {split}"):
            mask_path = split_masks_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                # Try other extensions
                for ext in ['.jpg', '.jpeg']:
                    if (split_masks_dir / (img_path.stem + ext)).exists():
                        mask_path = split_masks_dir / (img_path.stem + ext)
                        break
            
            if not mask_path.exists():
                continue
                
            split_source_images[split].add(img_path.name)
            
            try:
                extractor = PatchExtractor(str(img_path), str(mask_path))
                
                # Standard random patches
                rand_patches = extractor.get_random_patches()
                
                # Specific hard negative patches
                hard_negatives = extractor.get_hard_negative_patches(num_patches=4)
                hard_neg_count += len(hard_negatives)
                
                all_extracted = rand_patches + hard_negatives
                
                for patch_img, patch_mask in all_extracted:
                    oil_pct = compute_oil_percentage(patch_mask)
                    
                    if oil_pct >= OIL_THRESHOLD:
                        split_patches[1].append((patch_img, img_path.name))
                    elif oil_pct == 0:
                        split_patches[0].append((patch_img, img_path.name))
                    else:
                        # Ambiguous: 0 < oil_pct < OIL_THRESHOLD
                        ignored_count += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
        print(f"   Removed {ignored_count} ambiguous samples (0 < oil < {OIL_THRESHOLD})")
        print(f"   Added {hard_neg_count} hard negatives (texture/boundary)")
                
        # BALANCE THIS SPLIT (1:1)
        num_pos = len(split_patches[1])
        num_neg = len(split_patches[0])
        print(f"   Raw patches in {split}: Pos={num_pos}, Neg={num_neg}")
        
        target_count = min(num_pos, num_neg)
        balanced_patches = {
            1: random.sample(split_patches[1], target_count),
            0: random.sample(split_patches[0], target_count)
        }
        
        # Save patches for this split
        print(f"   Saving {target_count * 2} balanced patches to classification/{split}/...")
        for label in [0, 1]:
            label_dir = classification_root / split / str(label)
            label_dir.mkdir(parents=True, exist_ok=True)
            
            # Clear old patches
            for f in label_dir.glob("*.png"): f.unlink()
            
            for i, (patch, source_name) in enumerate(balanced_patches[label]):
                source_stem = Path(source_name).stem
                filename = f"patch_{i:06d}_{source_stem}.png"
                cv2.imwrite(str(label_dir / filename), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                
    # --- VERIFICATION ---
    print("\n" + "="*70)
    print("[VERIFY] DATA LEAKAGE VERIFICATION")
    print("="*70)
    
    train_imgs = split_source_images['train']
    val_imgs = split_source_images['val']
    test_imgs = split_source_images['test']
    
    print(f"Unique source images in TRAIN: {len(train_imgs)}")
    print(f"Unique source images in VAL:   {len(val_imgs)}")
    print(f"Unique source images in TEST:  {len(test_imgs)}")
    
    # Check for overlaps
    overlap_train_val = train_imgs.intersection(val_imgs)
    overlap_train_test = train_imgs.intersection(test_imgs)
    overlap_val_test = val_imgs.intersection(test_imgs)
    
    print(f"\nOverlap Train-Val:  {len(overlap_train_val)}")
    print(f"Overlap Train-Test: {len(overlap_train_test)}")
    print(f"Overlap Val-Test:   {len(overlap_val_test)}")
    
    assert len(overlap_train_val) == 0, "[FAIL] LEAKAGE DETECTED between Train and Val!"
    assert len(overlap_train_test) == 0, "[FAIL] LEAKAGE DETECTED between Train and Test!"
    assert len(overlap_val_test) == 0, "[FAIL] LEAKAGE DETECTED between Val and Test!"
    
    print("\n[OK] ZERO LEAKAGE VERIFIED: Each source image belongs to exactly one split.")
    print("="*70)

if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Find project root relative to this script
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    data_dir = project_root / "data"
    
    build_classification_dataset(data_dir)
