import streamlit as st
import pandas as pd
import numpy as np
plotly_available = True
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    plotly_available = False

from data_loader import fetch_financial_data
from features import engineer_features_and_labels
from model import run_walk_forward_backtest

st.set_page_config(
    page_title="QQQ Drawdown & Regime Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_and_process_data():
    df_raw = fetch_financial_data(start_date="2015-01-01")
    df_feat = engineer_features_and_labels(df_raw)
    test_dates, y_test, results, model, features = run_walk_forward_backtest(df_feat)
    return df_feat, test_dates, y_test, results, model, features

df_feat, test_dates, y_test, results, model, feature_names = load_and_process_data()

st.title("📉 QQQ 7% Drawdown Risk & Regime Dashboard")
st.markdown("""
**Research Question:** *Can a machine-learning model that uses market, volatility, and macro features predict whether QQQ will fall more than 7% below its current price at some point in the next 21 trading days, and does it provide a meaningful out-of-sample improvement over a simple volatility-only baseline?*
""")

# Sidebar Navigation
tab = st.sidebar.radio("Navigation", [
    "1. Live Risk Monitor", 
    "2. Model Performance & Baseline", 
    "3. Stress Period Analysis", 
    "4. Feature Importance & Limitations"
])

if tab == "1. Live Risk Monitor":
    st.header("Real-Time Drawdown Probability Monitor")
    latest_row = df_feat.iloc[-1]
    
    # Predict with trained model
    X_latest = pd.DataFrame([latest_row[feature_names]])
    current_prob = model.predict_proba(X_latest)[0][1]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current QQQ Price", f"${latest_row['QQQ']:.2f}")
    col2.metric("21-Day 7% Drawdown Probability", f"{current_prob * 100:.1f}%")
    col3.metric("VIX Index Level", f"{latest_row['VIX_Level']:.2f}")
    
    if current_prob > 0.40:
        st.error("🚨 **High Risk Alert:** Model indicates elevated probability of a >=7% drawdown in the next month. Consider hedging or reducing net exposure.")
    elif current_prob > 0.20:
        st.warning("⚠️ **Moderate Risk:** Volatility and market momentum suggest caution.")
    else:
        st.success("✅ **Low Risk:** Market indicators remain stable within normal historical bounds.")

elif tab == "2. Model Performance & Baseline":
    st.header("Out-of-Sample Baseline Comparison")
    st.markdown("Evaluating the **XGBoost Full Model** against the **Volatility-Only Baseline (Logistic Regression)** on out-of-sample test data.")
    
    metric_data = []
    for name, metrics in results.items():
        metric_data.append({
            "Model": name,
            "PR-AUC": f"{metrics['PR-AUC']:.4f}",
            "ROC-AUC": f"{metrics['ROC-AUC']:.4f}",
            "Brier Score": f"{metrics['Brier Score']:.4f}"
        })
    st.table(pd.DataFrame(metric_data))
    
    if plotly_available:
        st.subheader("Out-of-Sample Predicted Probabilities over Time")
        prob_df = pd.DataFrame({
            "Date": test_dates,
            "Full Model Prob": results["Full Model (XGBoost)"]["Probs"],
            "Baseline Prob": results["Baseline (Vol-Only)"]["Probs"],
            "Actual Drawdown": y_test.values
        }).set_index("Date")
        
        fig = px.line(prob_df, y=["Full Model Prob", "Baseline Prob"], title="Predicted Probability of 7% Drawdown (Next 21 Days)")
        st.plotly_chart(fig, use_container_width=True)

elif tab == "3. Stress Period Analysis":
    st.header("Performance Across Historical Stress Regimes")
    st.markdown("""
    Examining how the model behaves during major market dislocations:
    * **2020 COVID-19 Crash:** Rapid shock where volatility and volume spiked simultaneously.
    * **2022 Rate Hike / Inflation Cycle:** Prolonged drawdown with persistent yield curve shifts.
    """)
    
    # Filter test set around 2020 or 2022
    stress_period = df_feat.loc['2020-02-01':'2020-05-01']
    if plotly_available and not stress_period.empty:
        fig_stress = px.line(stress_period, y='QQQ', title="QQQ Price During 2020 COVID Stress Period")
        st.plotly_chart(fig_stress, use_container_width=True)
    else:
        st.write("Displaying historical volatility and drawdown metrics across tested stress windows.")

elif tab == "4. Feature Importance & Limitations":
    st.header("Model Feature Importance & Trustworthiness")
    
    if plotly_available:
        importances = model.feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=True)
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h", title="XGBoost Feature Importance")
        st.plotly_chart(fig_fi, use_container_width=True)
        
    st.markdown("""
    ### Key Limitations & Failure Modes
    1. **False Positives in Bull Markets:** During strong upward momentum, sudden short-lived pullbacks can trigger false alarms.
    2. **Transaction Costs & Hedging Friction:** Acting on every 40% probability signal could incur high turnover costs if hedging options prematurely.
    3. **Regime Shift Vulnerability:** Unprecedented macroeconomic shocks not present in the training distribution can degrade model calibration.
    """)