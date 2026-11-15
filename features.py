import numpy as np
import pandas as pd

def engineer_features_and_labels(df):
    """
    Computes features across 3 tiers (Volatility, Market/Momentum, Macro) 
    and defines the 21-day forward 7% drawdown binary target.
    """
    data = df.copy()
    
    # --- TARGET DEFINITION ---
    # 21 trading days forward window (~1 calendar month)
    # Target = 1 if min price in the next 21 days drops >= 7% below current price
    forward_window = 21
    future_min = data['QQQ'].shift(-1).rolling(window=forward_window).min()
    data['max_drawdown_forward'] = (future_min - data['QQQ']) / data['QQQ']
    data['Target'] = (data['max_drawdown_forward'] <= -0.07).astype(int)
    
    # --- TIER 1: VOLATILITY FEATURES ---
    data['Log_Return'] = np.log(data['QQQ'] / data['QQQ'].shift(1))
    data['Realized_Vol_5'] = data['Log_Return'].rolling(5).std() * np.sqrt(252)
    data['Realized_Vol_21'] = data['Log_Return'].rolling(21).std() * np.sqrt(252)
    data['Vol_Ratio'] = data['Realized_Vol_5'] / (data['Realized_Vol_21'] + 1e-8)
    data['VIX_Level'] = data['VIX']
    data['VIX_Change_5'] = data['VIX'].pct_change(5)
    
    # --- TIER 2: MARKET & MOMENTUM FEATURES ---
    data['SMA_50'] = data['QQQ'].rolling(50).mean()
    data['SMA_200'] = data['QQQ'].rolling(200).mean()
    data['Dist_SMA50'] = (data['QQQ'] - data['SMA_50']) / data['SMA_50']
    data['Dist_SMA200'] = (data['QQQ'] - data['SMA_200']) / data['SMA_200']
    
    # 14-day RSI
    delta = data['QQQ'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    data['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Volume trend
    data['Volume_Ratio'] = data['Volume'] / data['Volume'].rolling(50).mean()
    
    # --- TIER 3: MACRO / YIELD PROXIES ---
    data['TNX_Level'] = data['TNX']
    data['TNX_Change_21'] = data['TNX'].pct_change(21)
    
    # Drop rows with NaN from rolling calculations
    data = data.dropna()
    return data