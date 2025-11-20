import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator

def fetch_data(ticker, start_date, end_date):
    """
    Fetches historical data from yfinance.
    """
    # print(f"Fetching data for {ticker} from {start_date} to {end_date}...") # Commented out to be silent
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    
    print(f"✅ {ticker} downloaded")
    
    # Flatten MultiIndex columns if present (yfinance update)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df

def add_features(df):
    """
    Adds technical indicators to the dataframe.
    """
    df = df.copy()
    
    # Ensure we have data
    if len(df) < 50:
        return df

    # RSI
    rsi = RSIIndicator(close=df["Close"], window=14)
    df["RSI"] = rsi.rsi()

    # MACD
    macd = MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()

    # SMA
    df["SMA_20"] = SMAIndicator(close=df["Close"], window=20).sma_indicator()
    df["SMA_50"] = SMAIndicator(close=df["Close"], window=50).sma_indicator()
    
    # Returns
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    
    # Volatility
    df["Volatility"] = df["Log_Return"].rolling(window=20).std()

    return df

def prepare_data(ticker="SPY", start_date="2020-01-01", end_date="2024-01-01", lookback=30, split_percent=0.8):
    """
    Main function to load and prepare data for the model.
    Target: Next Day Open
    """
    df = fetch_data(ticker, start_date, end_date)
    df = add_features(df)
    
    # Target: Next Day Open
    df["Target"] = df["Open"].shift(-1)
    
    # Drop NaNs created by indicators and shifting
    df.dropna(inplace=True)
    
    feature_cols = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "MACD_Signal", "SMA_20", "SMA_50", "Log_Return", "Volatility"]
    
    # Split Data FIRST (before scaling) to avoid leakage
    split_idx = int(len(df) * split_percent)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    
    # Fit ONLY on training data
    X_train_scaled = scaler_X.fit_transform(train_df[feature_cols].values)
    # Transform test data using training statistics
    X_test_scaled = scaler_X.transform(test_df[feature_cols].values)
    
    scaler_y = StandardScaler()
    # Fit ONLY on training target
    y_train_scaled = scaler_y.fit_transform(train_df[["Target"]].values).flatten()
    # Transform test target
    y_test_scaled = scaler_y.transform(test_df[["Target"]].values).flatten()
    
    # Create sequences function
    def create_sequences(data, target, lookback):
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback])
            y.append(target[i+lookback])
        return np.array(X), np.array(y)

    # Create sequences for Train and Test separately
    # Note: For test, we might lose the first 'lookback' samples if we don't handle it carefully.
    # However, standard practice is to just sequence the available split data.
    # Ideally, we would use the last 'lookback' from train to start test, but for simplicity/strict separation:
    
    X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, lookback)
    X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, lookback)
    
    # Dates for test set (aligned with y_test)
    # The first prediction in test corresponds to lookback-th element in test_df
    dates_test = test_df.index[lookback:]
        
    return X_train, X_test, y_train, y_test, dates_test, feature_cols, scaler_y

def get_tickers_by_domain(domain, count=5):
    """
    Fetches a list of tickers belonging to a specific domain (Sector).
    Uses a predefined pool of major stocks to filter.
    """
    # Pool of major stocks across sectors
    STOCK_POOL = [
        # Technology
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "INTC", "CRM", "ADBE", "ORCL", "IBM", "QCOM", "CSCO",
        # Healthcare
        "JNJ", "LLY", "UNH", "MRK", "ABBV", "TMO", "PFE", "AMGN", "ISRG", "DHR", "BMY", "GILD", "CVS", "CI", "REGN",
        # Financial Services
        "JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "BLK", "C", "AXP", "SPGI", "CB", "PGR", "MMC", "SCHW",
        # Consumer Cyclical
        "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "LVS", "MAR", "HLT",
        # Communication Services
        "DIS", "CMCSA", "VZ", "T", "NFLX", "TMUS",
        # Industrials
        "CAT", "UNP", "GE", "HON", "UPS", "BA", "LMT", "RTX", "DE", "MMM"
    ]
    
    print(f"Searching for {count} stocks in '{domain}' sector...")
    matched_tickers = []
    
    for ticker in STOCK_POOL:
        if len(matched_tickers) >= count:
            break
            
        try:
            # Check sector
            info = yf.Ticker(ticker).info
            sector = info.get('sector', 'Unknown')
            
            if domain.lower() in sector.lower():
                matched_tickers.append(ticker)
                print(f"Found: {ticker} ({sector})")
        except Exception as e:
            print(f"Skipping {ticker}: {e}")
            continue
            
    if len(matched_tickers) < count:
        print(f"Warning: Only found {len(matched_tickers)} tickers for domain '{domain}'.")
        
    return matched_tickers
