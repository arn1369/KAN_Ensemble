import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import time

import yaml
from data_loader import prepare_data, get_tickers_by_domain
from model import KAN, BiLSTM_Attention
from ensemble import EnsembleManager

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def check_volatility(ticker, start_date, end_date):
    """
    Calculates annualized volatility for the given ticker.
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if df.empty:
            return 0.0
            
        # Calculate daily returns
        df['Return'] = df['Close'].pct_change()
        
        # Annualized Volatility (std dev of returns * sqrt(252))
        volatility = df['Return'].std() * np.sqrt(252)
        return volatility
    except Exception as e:
        # print(f"Error calculating volatility for {ticker}: {e}")
        return 0.0

def train_and_evaluate(ticker, config):
    print(f"\n{'='*50}")
    print(f"Processing {ticker}...")
    
    # 0. Volatility Check
    vol = check_volatility(ticker, config['data']['start_date'], config['data']['end_date'])
    VOL_THRESHOLD = config['volatility']['threshold']
    
    model_type = "KAN"
    if vol > VOL_THRESHOLD:
        model_type = "BiLSTM-Attn"
        print(f"⚠️ High Volatility Detected ({vol:.2%}). Switching to {model_type}.")
    else:
        print(f"✅ Stable Volatility ({vol:.2%}). Using {model_type}.")
        
    print(f"{'='*50}")
    
    # 1. Data Preparation
    LOOKBACK = config['data']['lookback']
    TRAINING_PERCENTAGE = config['training']['percentage']
    
    try:
        X_train, X_test, y_train, y_test, dates_test, feature_cols, scaler_y = prepare_data(
            ticker=ticker, 
            start_date=config['data']['start_date'],
            end_date=config['data']['end_date'],
            lookback=LOOKBACK,
            split_percent=TRAINING_PERCENTAGE
        )
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

    # Split Train/Test (Now handled inside prepare_data)
    # split_idx = int(len(X) * TRAINING_PERCENTAGE)
    # X_train, X_test = X[:split_idx], X[split_idx:]
    # y_train, y_test = y[:split_idx], y[split_idx:]
    # dates_test = dates[split_idx:]
    
    # 2. Model/Ensemble Initialization & Training
    input_dim_kan = X_train.shape[1] * X_train.shape[2] # Flattened
    input_dim_lstm = X_train.shape[2] # Features
    
    hidden_dim = config['model']['hidden_dim']
    output_dim = config['model']['output_dim']
    
    # Check if Ensemble is enabled
    if config.get('ensemble', {}).get('enabled', False):
        print(f"🚀 Starting Adaptive Ensemble Stream ({config['ensemble']['size']} models)...")
        
        # Initialize Ensemble Manager
        if model_type == "KAN":
            ensemble = EnsembleManager("KAN", input_dim_kan, hidden_dim, output_dim, config)
            # Flatten input for KAN training
            X_train_ens = X_train.reshape(X_train.shape[0], -1)
            X_test_ens = X_test.reshape(X_test.shape[0], -1)
        else:
            ensemble = EnsembleManager("BiLSTM-Attn", input_dim_lstm, hidden_dim, output_dim, config)
            X_train_ens = X_train
            X_test_ens = X_test
            
        # Train Ensemble
        ensemble.train_models(
            X_train_ens, y_train, 
            epochs=config['training']['epochs'], 
            lr=config['training']['learning_rate']
        )
        
        # Run Stream Simulation
        stream_results = ensemble.predict_stream(X_test_ens, y_test, scaler_y)
        
        # Metrics (based on Adaptive Prediction)
        predictions = np.array(stream_results['adaptive'])
        actuals = stream_results['actuals']
        
        rmse = np.sqrt(np.mean((predictions - actuals)**2))
        mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
        
        print(f"Stream MAPE (Adaptive): {mape:.2f}%")
        
        return {
            "ticker": ticker,
            "model": f"Ensemble-{model_type}",
            "volatility": vol,
            "rmse": rmse,
            "mape": mape,
            "dates": dates_test,
            "actuals": actuals,
            "predictions": predictions,
            "ensemble_mean": stream_results['ensemble_mean'],
            "best_expert": stream_results['best_expert'],
            "weights_history": stream_results['weights_history']
        }
        
    else:
        # SINGLE MODEL MODE (Legacy)
        # Convert to Tensor
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_test_tensor = torch.FloatTensor(X_test)
        
        if model_type == "KAN":
            model = KAN(input_dim_kan, hidden_dim, output_dim)
            X_train_tensor = X_train_tensor.view(X_train_tensor.shape[0], -1) # Flatten
            X_test_tensor = X_test_tensor.view(X_test_tensor.shape[0], -1)
        else:
            model = BiLSTM_Attention(
                input_dim_lstm, hidden_dim, output_dim, 
                num_layers=config['model']['lstm_layers'],
                dropout=config['model']['dropout']
            )
        
        # Training
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
        
        start_time = time.time()
        for epoch in range(config['training']['epochs']):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
        print(f"Training finished in {time.time() - start_time:.2f}s")
            
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
        
        predictions = scaler_y.inverse_transform(test_outputs.numpy())
        actuals = scaler_y.inverse_transform(y_test.reshape(-1, 1))
        
        rmse = np.sqrt(np.mean((predictions - actuals)**2))
        mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
        
        print(f"RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")
        
        return {
            "ticker": ticker,
            "model": model_type,
            "volatility": vol,
            "rmse": rmse,
            "mape": mape,
            "dates": dates_test,
            "actuals": actuals.flatten(),
            "predictions": predictions.flatten()
        }

def main():
    # Load Configuration
    config = load_config()
    print(f"Loaded Config: Domain={config['data']['domain']}, Count={config['data']['count']}")
    
    # Fetch Tickers dynamically
    TICKERS = get_tickers_by_domain(config['data']['domain'], config['data']['count'])
    
    if not TICKERS:
        print("No tickers found. Exiting.")
        return

    results = []
    
    # Adjust figure size dynamically based on count
    # If ensemble, we want 2 plots per ticker (Prediction + Weights)
    is_ensemble = config.get('ensemble', {}).get('enabled', False)
    
    if is_ensemble:
        rows = len(TICKERS)
        cols = 2
        plt.figure(figsize=(20, 5 * rows))
    else:
        rows = (len(TICKERS) + 2) // 3
        cols = 3
        plt.figure(figsize=(20, 5 * rows))
    
    for i, ticker in enumerate(TICKERS):
        res = train_and_evaluate(ticker, config)
        if res:
            results.append(res)
            
            if is_ensemble:
                # Plot 1: Predictions
                plt.subplot(rows, cols, i*2 + 1)
                plt.plot(res["dates"], res["actuals"], label="Actual", color="black", linewidth=1.5)
                plt.plot(res["dates"], res["ensemble_mean"], label="Ens Mean", color="blue", alpha=0.4, linestyle="--")
                plt.plot(res["dates"], res["predictions"], label="Adaptive", color="red", linewidth=1.5)
                plt.title(f"{ticker} Predictions (MAPE: {res['mape']:.2f}%)")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Plot 2: Weights Evolution
                plt.subplot(rows, cols, i*2 + 2)
                weights = res['weights_history'] # (steps, n_models)
                for model_idx in range(weights.shape[1]):
                    plt.plot(res["dates"], weights[:, model_idx], label=f"Model {model_idx+1}", alpha=0.6)
                plt.title(f"{ticker} Model Weights (Regime Switching)")
                plt.ylabel("Weight")
                plt.ylim(0, 1.0)
                # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.grid(True, alpha=0.3)
                
            else:
                # Standard Grid Plot
                plt.subplot(rows, cols, i+1)
                plt.plot(res["dates"], res["actuals"], label="Actual", color="black", linewidth=1.5)
                plt.plot(res["dates"], res["predictions"], label=f"Pred ({res['model']})", color="red", alpha=0.7)
                plt.title(f"{ticker} ({res['model']}) - MAPE: {res['mape']:.2f}%")
                plt.legend()
                plt.grid(True, alpha=0.3)
            
    plt.tight_layout()
    plt.savefig("ensemble_stream_results.png")
    print("\nEnsemble stream plot saved to ensemble_stream_results.png")
    
    print("\n" + "="*60)
    print(f"{'Ticker':<8} | {'Model':<15} | {'Vol':<8} | {'RMSE':<10} | {'MAPE':<10}")
    print("-" * 60)
    for res in results:
        print(f"{res['ticker']:<8} | {res['model']:<15} | {res['volatility']:<8.2%} | {res['rmse']:<10.4f} | {res['mape']:<10.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()
