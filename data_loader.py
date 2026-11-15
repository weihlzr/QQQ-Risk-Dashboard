import pandas as pd
import yfinance as yf
import time

def fetch_financial_data(start_date="2010-01-01", end_date=None, max_retries=3):
    """
    Downloads historical daily data for QQQ, VIX (^VIX), and 10-Year Treasury Yield (^TNX).
    Includes retry logic for handling database lock errors.
    """
    print("Downloading financial data from Yahoo Finance...")
    tickers = ["QQQ", "^VIX", "^TNX"]
    
    for attempt in range(max_retries):
        try:
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)
            break
        except Exception as e:
            if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                print(f"Database locked (attempt {attempt + 1}/{max_retries}). Waiting 2 seconds...")
                time.sleep(2)
            else:
                raise
    
    # Handle multi-index columns in modern yfinance
    if isinstance(data.columns, pd.MultiIndex):
        df = data['Close'].copy()
    else:
        df = data[['Close']].copy()
        
    df.columns = ['QQQ', 'VIX', 'TNX']
    df = df.ffill().dropna()
    
    # Download QQQ Volume as well
    for attempt in range(max_retries):
        try:
            qqq_vol = yf.download("QQQ", start=start_date, end=end_date, progress=False)
            break
        except Exception as e:
            if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                print(f"Database locked (attempt {attempt + 1}/{max_retries}). Waiting 2 seconds...")
                time.sleep(2)
            else:
                raise
                
    if isinstance(qqq_vol.columns, pd.MultiIndex):
        df['Volume'] = qqq_vol['Volume'].iloc[:, 0]
    else:
        df['Volume'] = qqq_vol['Volume']
        
    df = df.ffill().dropna()
    
    if df.empty:
        raise ValueError("Downloaded data is empty. Check ticker symbols or date range.")
    
    print(f"Successfully downloaded data: {len(df)} rows")
    return df