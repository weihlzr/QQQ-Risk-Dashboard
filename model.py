import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

def run_walk_forward_backtest(df):
    """
    Performs walk-forward validation comparing a Volatility-only Baseline 
    against the Full Feature XGBoost Model.
    """
    feature_cols_baseline = ['Realized_Vol_21', 'VIX_Level']
    feature_cols_full = [
        'Realized_Vol_5', 'Realized_Vol_21', 'Vol_Ratio', 'VIX_Level', 'VIX_Change_5',
        'Dist_SMA50', 'Dist_SMA200', 'RSI_14', 'Volume_Ratio', 'TNX_Level', 'TNX_Change_21'
    ]
    target_col = 'Target'
    
    if df.empty:
        raise ValueError("Input dataframe is empty. Cannot run backtest.")
    
    # Check that all required features exist
    missing_cols = set(feature_cols_baseline + feature_cols_full + [target_col]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Split into Train (approx 60%), Val/Test (40%)
    train_size = int(len(df) * 0.6)
    
    if train_size < 1:
        raise ValueError("Dataset too small for train/test split. Need at least 2 rows.")
    
    X_train_base = df[feature_cols_baseline].iloc[:train_size]
    X_test_base = df[feature_cols_baseline].iloc[train_size:]
    
    X_train_full = df[feature_cols_full].iloc[:train_size]
    X_test_full = df[feature_cols_full].iloc[train_size:]
    
    y_train = df[target_col].iloc[:train_size]
    y_test = df[target_col].iloc[train_size:]
    
    if len(X_train_base) == 0 or len(y_train) == 0:
        raise ValueError(f"Training set is empty. Train size: {len(X_train_base)}, y_train size: {len(y_train)}")
    if len(X_test_base) == 0 or len(y_test) == 0:
        raise ValueError(f"Test set is empty. Test size: {len(X_test_base)}, y_test size: {len(y_test)}")
    
    # 1. Baseline Model (Logistic Regression)
    base_model = LogisticRegression(class_weight='balanced', random_state=42)
    base_model.fit(X_train_base, y_train)
    base_probs = base_model.predict_proba(X_test_base)[:, 1]
    
    # 2. Full Model (XGBoost)
    scale_pos_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-8)
    full_model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'
    )
    full_model.fit(X_train_full, y_train)
    full_probs = full_model.predict_proba(X_test_full)[:, 1]
    
    # Evaluation Metrics
    results = {}
    
    for name, probs in [("Baseline (Vol-Only)", base_probs), ("Full Model (XGBoost)", full_probs)]:
        precision, recall, _ = precision_recall_curve(y_test, probs)
        pr_auc = auc(recall, precision)
        roc_auc = roc_auc_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)
        results[name] = {
            "PR-AUC": pr_auc,
            "ROC-AUC": roc_auc,
            "Brier Score": brier,
            "Probs": probs
        }
        
    return df.index[train_size:], y_test, results, full_model, feature_cols_full