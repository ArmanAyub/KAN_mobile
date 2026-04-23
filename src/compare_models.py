import torch
from model import BaselineClassifier, KANClassifier
from utils import get_model_summary

def compare_complexity():
    # 1. Initialize models
    baseline = BaselineClassifier(num_classes=2)
    kan = KANClassifier(num_classes=2, hidden_dim=64)
    
    # 2. Get summaries
    b_total, b_train = get_model_summary(baseline)
    k_total, k_train = get_model_summary(kan)
    
    # 3. Print Comparison
    print("\n--- MODEL COMPLEXITY COMPARISON ---")
    print(f"{'Metric':<20} | {'Baseline (MLP)':<20} | {'FastKAN Head':<20}")
    print("-" * 65)
    print(f"{'Total Parameters':<20} | {b_total:<20,} | {k_total:<20,}")
    print(f"{'Trainable Params':<20} | {b_train:<20,} | {k_train:<20,}")
    
    # 4. Savings Calculation
    diff = b_train - k_train
    percent = (diff / b_train) * 100
    if diff > 0:
        print(f"\nKAN head uses {diff:,} fewer trainable parameters ({percent:.2f}% reduction)!")
    else:
        print(f"\nBaseline head uses {-diff:,} fewer trainable parameters ({-percent:.2f}% reduction)!")

if __name__ == "__main__":
    compare_complexity()
