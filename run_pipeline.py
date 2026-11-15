import os
import pandas as pd
from data_loader import fetch_financial_data
from features import engineer_features_and_labels
from model import run_walk_forward_backtest

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df_raw = fetch_financial_data(start_date="2012-01-01")
    
    if df_raw.empty:
        print("ERROR: Failed to download financial data.")
        exit(1)
    
    df_processed = engineer_features_and_labels(df_raw)
    
    print(f"Processed dataset shape: {df_processed.shape}")
    print(f"Positive drawdown instances (7% drop in 21 days): {df_processed['Target'].sum()} out of {len(df_processed)}")
    
    if df_processed.empty:
        print("ERROR: Processed dataset is empty. Check feature engineering logic.")
        exit(1)
    
    if df_processed['Target'].sum() == 0:
        print("WARNING: No positive examples (drawdown events) in the dataset.")
    
    test_dates, y_test, results, model, features = run_walk_forward_backtest(df_processed)
    print("\n--- Out-of-Sample Results ---")
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  PR-AUC      : {metrics['PR-AUC']:.4f}")
        print(f"  ROC-AUC     : {metrics['ROC-AUC']:.4f}")
        print(f"  Brier Score : {metrics['Brier Score']:.4f}")