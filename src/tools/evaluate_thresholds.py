"""
Evaluate different classification thresholds to find the best trade-off between Recall and Precision.
Usage: python src/tools/evaluate_thresholds.py --csv outputs/classification/val_predictions_epoch_1.csv
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from pathlib import Path

def evaluate_thresholds(csv_path):
    # Load CSV
    df = pd.read_csv(csv_path)
    
    y_true = df['true_label'].values
    y_probs = df['predicted_probability'].values
    
    thresholds = [0.5, 0.4, 0.35, 0.3, 0.25, 0.2]
    
    results = []
    
    print("\n" + "="*80)
    print(f"{'Threshold':<10} | {'TP':<5} | {'TN':<5} | {'FP':<5} | {'FN':<5} | {'Prec':<7} | {'Rec':<7} | {'F1':<7}")
    print("-" * 80)
    
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results.append({
            'threshold': t,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })
        
        print(f"{t:<10.2f} | {tp:<5} | {tn:<5} | {fp:<5} | {fn:<5} | {prec:<7.4f} | {rec:<7.4f} | {f1:<7.4f}")
    
    print("="*80)
    
    # Recommendation logic: Maximize Recall while keeping FP < 15% of total negatives
    total_neg = np.sum(y_true == 0)
    valid_results = [r for r in results if r['fp'] < 0.15 * total_neg]
    
    if valid_results:
        best = max(valid_results, key=lambda x: x['recall'])
    else:
        best = max(results, key=lambda x: x['f1'])
        
    print(f"\nRecommended Threshold: {best['threshold']:.2f}")
    print(f"Reason: Recall of {best['recall']:.2f} with reasonable False Positives ({best['fp']}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate classification thresholds.")
    parser.add_argument("--csv", type=str, required=True, help="Path to prediction CSV.")
    
    args = parser.parse_args()
    
    csv_file = Path(args.csv)
    if not csv_file.exists():
        print(f"Error: CSV file not found at {csv_file}")
    else:
        evaluate_thresholds(csv_file)
