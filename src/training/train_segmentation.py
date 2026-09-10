from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.utils import SegmentationSplitDataset, seed_everything


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 12
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    image_size: int = 224
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum(dim=(1, 2, 3))
        union = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_value = self.bce(logits, targets)
        dice_value = self.dice(logits, targets)
        return self.bce_weight * bce_value + (1.0 - self.bce_weight) * dice_value


from src.models.segmentation import build_segmentation_model


def binary_scores(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    predictions = (torch.sigmoid(logits) > 0.5).float()
    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    union = predictions.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (predictions.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1e-6)
    return float(iou.mean().item()), float(dice.mean().item())


def run_epoch(model, loader, criterion, optimizer, device):
    training = optimizer is not None
    model.train(training)
    losses = []
    ious = []
    dices = []

    for images, masks, _, _ in loader:
        images = images.to(device)
        masks = masks.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, masks)

        if training:
            loss.backward()
            optimizer.step()

        iou, dice = binary_scores(logits.detach(), masks)
        losses.append(loss.item())
        ious.append(iou)
        dices.append(dice)

    return (
        float(np.mean(losses)) if losses else 0.0,
        float(np.mean(ious)) if ious else 0.0,
        float(np.mean(dices)) if dices else 0.0,
    )


def denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image_tensor.cpu() * std + mean).clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def save_prediction_grid(samples: list[dict], output_path: Path) -> None:
    rows = max(len(samples), 1)
    figure, axes = plt.subplots(rows, 3, figsize=(12, 4 * rows))
    if rows == 1:
        axes = np.array([axes])

    for row_axes in axes:
        for axis in row_axes:
            axis.axis("off")

    for row_index, sample in enumerate(samples):
        axes[row_index, 0].imshow(denormalize(sample["image"]))
        axes[row_index, 0].set_title("Image", fontsize=9)
        axes[row_index, 1].imshow(sample["mask"].squeeze(0).cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row_index, 1].set_title("Ground Truth", fontsize=9)
        axes[row_index, 2].imshow(sample["prediction"].squeeze(0).cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row_index, 2].set_title("Prediction", fontsize=9)

    plt.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def evaluate(model, loader, device, output_root: Path) -> dict:
    model.eval()
    ious = []
    dices = []
    samples = []

    with torch.no_grad():
        for images, masks, _, _ in loader:
            logits = model(images.to(device))
            predictions = (torch.sigmoid(logits) > 0.5).float().cpu()
            batch_iou, batch_dice = binary_scores(logits, masks.to(device))
            ious.append(batch_iou)
            dices.append(batch_dice)

            if len(samples) < 5:
                for index in range(images.size(0)):
                    if len(samples) >= 5:
                        break
                    samples.append(
                        {
                            "image": images[index],
                            "mask": masks[index],
                            "prediction": predictions[index],
                        }
                    )

    visualization_path = output_root / "segmentation_predictions.png"
    save_prediction_grid(samples, visualization_path)
    return {
        "iou": round(float(np.mean(ious)) if ious else 0.0, 4),
        "dice": round(float(np.mean(dices)) if dices else 0.0, 4),
        "sample_visualization": str(visualization_path),
    }


def plot_segmentation_curves(history: list[dict], output_root: Path) -> None:
    """Generate and save Dice and Loss curves for segmentation training."""
    epochs = [h['epoch'] for h in history]
    train_dice = [h['train_dice'] for h in history]
    val_dice = [h['val_dice'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]

    plt.figure(figsize=(15, 5))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title('Segmentation Loss (BCE + Dice)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Dice curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_dice, label='Train Dice')
    plt.plot(epochs, val_dice, label='Val Dice')
    plt.title('Segmentation Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_root / "training_curves.png")
    plt.close()


def main() -> None:
    from src.data.utils import seed_everything
    seed_everything(42)
    
    project_root = PROJECT_ROOT
    split_root = project_root / "data" / "splits" / "segmentation"
    output_root = project_root / "outputs" / "segmentation"
    output_root.mkdir(parents=True, exist_ok=True)
    
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    config = TrainingConfig()
    device = torch.device(config.device)

    print("="*70)
    print("TRAINING SEGMENTATION MODEL")
    print("="*70)
    print(f"Device: {device}")

    train_dataset = SegmentationSplitDataset(split_root, "train", train=True)
    val_dataset = SegmentationSplitDataset(split_root, "val", train=False)
    test_dataset = SegmentationSplitDataset(split_root, "test", train=False)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = build_segmentation_model().to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    history = []
    best_val_dice = -1.0
    best_model_path = models_dir / "segmentation_best.pth"
    last_model_path = models_dir / "segmentation_last.pth"

    for epoch in range(1, config.epochs + 1):
        print(f"\n[Epoch {epoch:2d}/{config.epochs}]")
        train_loss, train_iou, train_dice = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou, val_dice = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | Train IoU: {train_iou:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Dice:   {val_dice:.4f} | Val IoU:   {val_iou:.4f}")

        epoch_result = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "train_iou": round(train_iou, 4),
            "val_iou": round(val_iou, 4),
            "train_dice": round(train_dice, 4),
            "val_dice": round(val_dice, 4),
        }
        history.append(epoch_result)

        # Save best model based on Dice
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": asdict(config),
                "epoch": epoch,
                "best_metric": best_val_dice
            }, best_model_path)
            print(f"  --> Best model saved (Dice: {val_dice:.4f})")
            
        # Save last model
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "final_metric": val_dice
        }, last_model_path)

    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate(model, test_loader, device, output_root)

    # Generate Curves
    plot_segmentation_curves(history, output_root)
    print(f"Training curves saved to {output_root / 'training_curves.png'}")

    # Save Detailed History
    history_file = output_root / "history.json"
    history_file.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"History saved to {history_file}")

    # Save Final Summary
    best_epoch = int(np.argmax([h['val_dice'] for h in history]) + 1)
    summary = {
        "best_epoch": best_epoch,
        "best_dice": float(best_val_dice),
        "final_dice": float(history[-1]['val_dice']),
        "test_iou": float(test_metrics['iou']),
        "test_dice": float(test_metrics['dice'])
    }
    summary_file = output_root / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary saved to {summary_file}")

    # Final Summary Print
    print("\n" + "="*70)
    print("FINAL TRAINING SUMMARY")
    print("="*70)
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Dice: {best_val_dice:.4f}")
    print(f"Final Val Dice: {history[-1]['val_dice']:.4f}")
    print("="*70)
    
    print("\n[OK] SEGMENTATION TRAINING COMPLETE")


if __name__ == "__main__":
    main()
