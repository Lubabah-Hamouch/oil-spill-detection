"""
Visualize classification results from a prediction CSV file.
Usage: python src/tools/visualize_classification.py --csv outputs/classification/val_predictions_epoch_1.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import random

def visualize_predictions(csv_path, num_samples=20):
    # Load CSV
    df = pd.read_csv(csv_path)
    
    if len(df) < num_samples:
        num_samples = len(df)
        
    # Sample random examples
    sample_df = df.sample(n=num_samples).reset_index(drop=True)
    
    # Create grid (4x5)
    rows, cols = 4, 5
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
    axes = axes.flatten()
    
    label_map = {0: 'No Spill', 1: 'Spill'}
    
    for i in range(num_samples):
        row = sample_df.iloc[i]
        img_path = row['image_path']
        true_label = int(row['true_label'])
        pred_label = int(row['predicted_label'])
        prob = float(row['predicted_probability'])
        
        # Determine correctness
        is_correct = (true_label == pred_label)
        color = 'green' if is_correct else 'red'
        
        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
            axes[i].imshow(img)
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error loading:\n{Path(img_path).name}", 
                         ha='center', va='center')
            print(f"Error loading {img_path}: {e}")
            
        # Set title
        title = (f"True: {label_map[true_label]}\n"
                 f"Pred: {label_map[pred_label]}\n"
                 f"Prob: {prob:.4f}")
        
        # Highlight FP/FN in title
        if not is_correct:
            error_type = "FP" if pred_label == 1 else "FN"
            title += f" ({error_type})"
            
        axes[i].set_title(title, color=color, fontsize=10, fontweight='bold')
        axes[i].axis('off')
        
    # Hide empty axes if num_samples < rows*cols
    for j in range(num_samples, rows * cols):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize classification predictions.")
    parser.add_argument("--csv", type=str, required=True, help="Path to the prediction CSV file.")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to visualize.")
    
    args = parser.parse_args()
    
    csv_file = Path(args.csv)
    if not csv_file.exists():
        print(f"Error: CSV file not found at {csv_file}")
    else:
        visualize_predictions(csv_file, args.samples)
