import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model import KAN, BiLSTM_Attention

class EnsembleManager:
    def __init__(self, model_type, input_dim, hidden_dim, output_dim, config):
        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.config = config
        self.size = config['ensemble']['size']
        self.window = config['ensemble']['window']
        self.models = []
        self.weights = np.ones(self.size) / self.size # Initial equal weights
        self.errors = [[] for _ in range(self.size)] # Track errors for each model
        
        self._initialize_models()
        
    def _initialize_models(self):
        """Initializes N models with different random seeds."""
        for i in range(self.size):
            torch.manual_seed(i + 42) # Different seed for each model
            if self.model_type == "KAN":
                model = KAN(self.input_dim, self.hidden_dim, self.output_dim)
            else:
                # For LSTM, input_dim is features (not flattened)
                # We need to handle this distinction carefully in forward pass
                model = BiLSTM_Attention(
                    self.input_dim, 
                    self.hidden_dim, 
                    self.output_dim, 
                    num_layers=self.config['model']['lstm_layers'],
                    dropout=self.config['model']['dropout']
                )
            self.models.append(model)
            
    def train_models(self, X_train, y_train, epochs=50, lr=0.001):
        """Trains all models in the ensemble."""
        print(f"Training ensemble of {self.size} {self.model_type} models...")
        
        criterion = nn.MSELoss()
        
        for i, model in enumerate(self.models):
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            model.train()
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X_train)
            y_tensor = torch.FloatTensor(y_train).unsqueeze(1)
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                
            # print(f"Model {i+1}/{self.size} trained.")
            
    def predict_stream(self, X_test, y_test, scaler_y):
        """
        Simulates a stream of data.
        For each time step:
        1. Make predictions with all models.
        2. Calculate ensemble predictions (Mean, Adaptive).
        3. Observe actual value.
        4. Update weights based on error.
        """
        predictions_ensemble_mean = []
        predictions_adaptive = []
        predictions_best_expert = []
        
        actuals = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # History of individual model predictions (inverse transformed)
        model_preds_history = [[] for _ in range(self.size)]
        
        # Track weights over time for visualization
        weights_history_list = []
        
        print("Starting stream simulation...")
        
        for t in range(len(X_test)):
            # Record current weights BEFORE update (used for this prediction)
            weights_history_list.append(self.weights.copy())
            
            # Current input
            x_t = torch.FloatTensor(X_test[t]).unsqueeze(0) # (1, seq, feat) or (1, flat)
            
            # 1. Get predictions from all models
            current_preds = []
            for i, model in enumerate(self.models):
                model.eval()
                with torch.no_grad():
                    pred = model(x_t).item()
                    # Inverse transform immediately for weighting logic
                    pred_inv = scaler_y.inverse_transform([[pred]])[0][0]
                    current_preds.append(pred_inv)
                    model_preds_history[i].append(pred_inv)
            
            current_preds = np.array(current_preds)
            
            # 2. Ensemble Predictions
            
            # A. Simple Mean
            ens_mean = np.mean(current_preds)
            predictions_ensemble_mean.append(ens_mean)
            
            # B. Adaptive Weighted Average
            adaptive_pred = np.sum(current_preds * self.weights)
            predictions_adaptive.append(adaptive_pred)
            
            # C. Best Expert (Model with highest weight)
            best_idx = np.argmax(self.weights)
            predictions_best_expert.append(current_preds[best_idx])
            
            # 3. Observe Actual
            actual = actuals[t]
            
            # 4. Update Weights
            self._update_weights(current_preds, actual)
            
        return {
            "actuals": actuals,
            "ensemble_mean": predictions_ensemble_mean,
            "adaptive": predictions_adaptive,
            "best_expert": predictions_best_expert,
            "weights_history": np.array(weights_history_list) # Shape: (steps, n_models)
        }
        
    def _update_weights(self, predictions, actual):
        """
        Updates model weights based on recent errors (Inverse Variance / Exponential).
        """
        # Calculate squared error for this step
        step_errors = (predictions - actual) ** 2
        
        # Update error history
        for i in range(self.size):
            self.errors[i].append(step_errors[i])
            # Keep only recent window
            if len(self.errors[i]) > self.window:
                self.errors[i].pop(0)
                
        # Calculate Mean Squared Error over window
        window_mses = np.array([np.mean(errs) for errs in self.errors])
        
        # Avoid division by zero
        window_mses = np.maximum(window_mses, 1e-6)
        
        # Inverse Variance Weighting (Lower error -> Higher weight)
        inv_errors = 1.0 / window_mses
        new_weights = inv_errors / np.sum(inv_errors)
        
        # Soft update (Exponential Moving Average of weights) to prevent jumping
        alpha = 0.2 # Smoothing factor
        self.weights = (1 - alpha) * self.weights + alpha * new_weights
