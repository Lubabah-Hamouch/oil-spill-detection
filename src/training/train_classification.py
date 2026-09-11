"""
Train classification model on balanced patch dataset.
- MobileNetV3_Large with pretrained weights
- CrossEntropyLoss (no class weights - dataset is balanced)
- Proper train/val/test evaluation
- Detailed logging and visualization
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_large
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "splits" / "classification"


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 32
    lr: float = 0.001
    weight_decay: float = 1e-5
    num_workers: int = 0
    image_size: int = 224
    threshold: float = 0.20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ClassificationDataset(Dataset):
    """Custom dataset for classification patches."""
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.source_ids = []
        
        # Load paths for each class
        for label in [0, 1]:
            class_dir = self.root_dir / split / str(label)
            if not class_dir.exists():
                raise FileNotFoundError(f"Dataset directory not found: {class_dir}")
            
            image_files = sorted(list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg")))
            self.image_paths.extend(image_files)
            self.labels.extend([label] * len(image_files))
            
            # Extract source_id from filename: patch_idx_sourceid.png
            for path in image_files:
                parts = path.stem.split('_')
                if len(parts) >= 3:
                    source_id = "_".join(parts[2:]) # Extract source name
                else:
                    source_id = "unknown"
                self.source_ids.append(source_id)
        
        print(f"Loaded {len(self.image_paths)} {split} samples")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        source_id = self.source_ids[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label, str(image_path), source_id


def get_transforms(split='train'):
    """Get data transforms for train/val/test."""
    
    if split == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


from src.models.classification import build_classification_model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for images, labels, _, _ in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Compute metrics
    avg_loss = running_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device, split='val'):
    """Evaluate model on a given dataset split with robust metrics."""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []
    all_paths = []
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=f"Evaluating {split}", leave=False)
        
        for images, labels, paths, _ in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            loss = criterion(outputs, labels.long())
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Compute metrics
    avg_loss = running_loss / len(data_loader)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)
    
    # 3. PROTECT CONFUSION MATRIX
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    
    # 4. MAKE ROC-AUC ROBUST
    roc_auc = None
    if len(np.unique(all_labels)) > 1:
        try:
            roc_auc = roc_auc_score(all_labels, all_probs)
        except:
            roc_auc = None
    
    # 5. CLASS DISTRIBUTION DEBUG
    num_pos = np.sum(all_labels == 1)
    num_neg = np.sum(all_labels == 0)
    print(f"  {split} Distribution: Pos: {num_pos}, Neg: {num_neg}")
    
    # 6. PREDICTION COLLAPSE CHECK
    num_pred_pos = np.sum(all_preds == 1)
    num_pred_neg = np.sum(all_preds == 0)
    print(f"  Predictions: Pos: {num_pred_pos}, Neg: {num_pred_neg}")
    
    # 5. ADD MEAN PROBABILITY DEBUG (IMPORTANT SIGNAL)
    print(f"  Mean Prob (pos class): {np.mean(all_probs):.4f}")
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'labels': all_labels,
        'predictions': all_preds,
        'probabilities': all_probs,
        'paths': all_paths
    }


def plot_training_curves(history, output_dir):
    """Plot training/validation curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    axes[0].plot(history['train_loss'], label='Training Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Validation Loss', marker='s')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(history['train_accuracy'], label='Training Accuracy', marker='o')
    axes[1].plot(history['val_accuracy'], label='Validation Accuracy', marker='s')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'training_curves.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {output_file}")
    plt.close()


def plot_confusion_matrix(cm, output_dir):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Spill', 'Spill'],
                yticklabels=['No Spill', 'Spill'],
                ax=ax)
    
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / 'confusion_matrix.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to {output_file}")
    plt.close()


def verify_leakage(train_ds, val_ds, test_ds):
    """Verify that source images are unique per split using explicit source_ids."""
    print("\n" + "="*70)
    print("RUNNING FINAL LEAKAGE SANITY CHECK")
    print("="*70)
    
    train_sources = set(train_ds.source_ids)
    val_sources = set(val_ds.source_ids)
    test_sources = set(test_ds.source_ids)
    
    print(f"  Train unique sources: {len(train_sources)}")
    print(f"  Val unique sources:   {len(val_sources)}")
    print(f"  Test unique sources:  {len(test_sources)}")
    
    # Overlaps
    o1 = train_sources.intersection(val_sources)
    o2 = train_sources.intersection(test_sources)
    o3 = val_sources.intersection(test_sources)
    
    print(f"\n  Overlaps: Train-Val: {len(o1)}, Train-Test: {len(o2)}, Val-Test: {len(o3)}")
    
    # Random sample check
    if len(val_ds) > 0:
        _, _, path, sid = val_ds[0]
        print(f"\n  Checking random val sample: {Path(path).name}")
        print(f"  Source ID: {sid}")
        if sid in train_sources:
            print(f"  [ERROR] FOUND in train split → leakage!")
        else:
            print(f"  [OK] not found in train split")

    assert len(o1) == 0 and len(o2) == 0 and len(o3) == 0, "DATA LEAKAGE DETECTED!"
    print("  [SUCCESS] No leakage detected across splits.")
    print("="*70 + "\n")


def save_predictions(metrics, epoch, output_dir):
    """Save validation predictions to a CSV file for manual inspection."""
    df = pd.DataFrame({
        'image_path': [str(p) for p in metrics['paths']],
        'true_label': metrics['labels'],
        'predicted_probability': metrics['probabilities'],
        'predicted_label': metrics['predictions']
    })
    
    csv_dir = output_dir
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = csv_dir / f"val_predictions_epoch_{epoch}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Validation predictions saved to {csv_path}")


def main() -> None:
    """Main training function."""
    from src.data.utils import seed_everything
    seed_everything(42)
    
    config = TrainingConfig()
    device = torch.device(config.device)
    
    project_root = PROJECT_ROOT
    output_dir = project_root / "outputs" / "classification"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("TRAINING CLASSIFICATION MODEL")
    print("="*70)
    print(f"Device: {device}")
    
    # Load datasets
    print("\nLoading datasets...")
    
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    test_transform = get_transforms('test')
    
    train_dataset = ClassificationDataset(DATA_DIR, split='train', transform=train_transform)
    val_dataset = ClassificationDataset(DATA_DIR, split='val', transform=val_transform)
    test_dataset = ClassificationDataset(DATA_DIR, split='test', transform=test_transform)
    
    # Verify leakage before starting
    verify_leakage(train_dataset, val_dataset, test_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    # Build model
    print("\nBuilding model...")
    model = build_classification_model(num_classes=2, pretrained=True).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    history = defaultdict(list)
    best_val_f1 = -1.0
    best_model_path = models_dir / "classification_best.pth"
    last_model_path = models_dir / "classification_last.pth"
    
    start_time = time.time()
    
    for epoch in range(config.epochs):
        print(f"\n[Epoch {epoch+1:2d}/{config.epochs}]")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, split='val')
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_roc_auc'].append(val_metrics['roc_auc'])
        
        # Save validation predictions to CSV
        save_predictions(val_metrics, epoch + 1, output_dir)
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Logging
        roc_auc_val = val_metrics['roc_auc']
        roc_auc_str = f"{roc_auc_val:.4f}" if roc_auc_val is not None else "N/A"
        
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")
        print(f"  F1: {val_metrics['f1']:.4f} | ROC-AUC: {roc_auc_str}")
        print(f"  Confusion Matrix: {val_metrics['confusion_matrix'].tolist()}")
        
        # Save best model based on F1-score
        current_f1 = val_metrics['f1']
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": asdict(config),
                "epoch": epoch + 1,
                "best_metric": best_val_f1
            }, best_model_path)
            print(f"  --> Best model saved (F1: {current_f1:.4f})")
        
        # Save last model
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "epoch": epoch + 1,
            "final_metric": current_f1
        }, last_model_path)
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.1f} minutes")
    
    # Load best model for testing
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    
    checkpoint = torch.load(best_model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate(model, test_loader, criterion, device, split='test')
    
    roc_auc_test = test_metrics['roc_auc']
    roc_auc_test_str = f"{roc_auc_test:.4f}" if roc_auc_test is not None else "N/A"
    
    print(f"\nTest Set Metrics:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {roc_auc_test_str}")
    
    print(f"\nConfusion Matrix:")
    print(f"  {test_metrics['confusion_matrix']}")
    
    # Per-class metrics
    print(f"\nPer-class Report:")
    report = classification_report(
        test_metrics['labels'],
        test_metrics['predictions'],
        target_names=['No Spill', 'Spill']
    )
    print(report)
    
    # Visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    plot_training_curves(history, output_dir)
    plot_confusion_matrix(test_metrics['confusion_matrix'], output_dir)
    
    # Save Detailed History
    history_file = output_dir / "history.json"
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {history_file}")

    # Save Final Summary
    best_epoch = int(np.argmax(history['val_f1']) + 1)
    summary = {
        "best_epoch": best_epoch,
        "best_f1": float(best_val_f1),
        "final_f1": float(history['val_f1'][-1]),
        "test_accuracy": float(test_metrics['accuracy']),
        "test_f1": float(test_metrics['f1'])
    }
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_file}")

    # Final Summary Print
    print("\n" + "="*70)
    print("FINAL TRAINING SUMMARY")
    print("="*70)
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val F1: {best_val_f1:.4f}")
    print(f"Final Val F1: {history['val_f1'][-1]:.4f}")
    print("="*70)
    
    print("\n[OK] CLASSIFICATION TRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
