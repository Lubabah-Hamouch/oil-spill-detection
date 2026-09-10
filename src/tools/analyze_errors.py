"""
Analyze classification errors and categorize images into TP, TN, FP, FN.
Usage: python src/tools/analyze_errors.py --csv outputs/classification/val_predictions_epoch_1.csv
"""

import argparse
import pandas as pd
import shutil
from pathlib import Path
import random

def analyze_errors(csv_path, output_root, samples_per_cat=10):
    # Load CSV
    df = pd.read_csv(csv_path)
    total = len(df)
    
    # Define categories
    # TP: True=1, Pred=1
    # TN: True=0, Pred=0
    # FP: True=0, Pred=1 (False Alarm)
    # FN: True=1, Pred=0 (Missed)
    
    categories = {
        'TP': df[(df['true_label'] == 1) & (df['predicted_label'] == 1)],
        'TN': df[(df['true_label'] == 0) & (df['predicted_label'] == 0)],
        'FP': df[(df['true_label'] == 0) & (df['predicted_label'] == 1)],
        'FN': df[(df['true_label'] == 1) & (df['predicted_label'] == 0)]
    }
    
    print("\n" + "="*50)
    print("CLASSIFICATION ERROR ANALYSIS")
    print("="*50)
    print(f"Total validation samples: {total}")
    print("-"*50)
    
    output_root = Path(output_root)
    
    for cat_name, cat_df in categories.items():
        count = len(cat_df)
        pct = (count / total) * 100 if total > 0 else 0
        print(f"{cat_name:3s}: {count:5d} ({pct:6.2f}%)")
        
        # Prepare directory
        cat_dir = output_root / cat_name
        if cat_dir.exists():
            shutil.rmtree(cat_dir)
        cat_dir.mkdir(parents=True, exist_ok=True)
        
        # Select samples
        if count > 0:
            num_to_save = min(count, samples_per_cat)
            samples = cat_df.sample(n=num_to_save)
            
            for i, (_, row) in enumerate(samples.iterrows()):
                src_path = Path(row['image_path'])
                if src_path.exists():
                    # Name includes prob for easier debugging
                    dst_name = f"sample_{i:02d}_prob_{row['predicted_probability']:.4f}_{src_path.name}"
                    shutil.copy(src_path, cat_dir / dst_name)
    
    print("-"*50)
    print(f"Sample images saved to: {output_root}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze classification errors.")
    parser.add_argument("--csv", type=str, required=True, help="Path to prediction CSV.")
    parser.add_argument("--out", type=str, default="outputs/debug", help="Output directory for categorized images.")
    parser.add_argument("--samples", type=int, default=10, help="Samples to save per category.")
    
    args = parser.parse_args()
    
    csv_file = Path(args.csv)
    if not csv_file.exists():
        print(f"Error: CSV file not found at {csv_file}")
    else:
        analyze_errors(csv_file, args.out, args.samples)
