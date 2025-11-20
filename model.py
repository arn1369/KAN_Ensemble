import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network Layer.
    Replaces linear weights with learnable spline functions.
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Base weight (like residual connection)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Spline weights
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        
        # Learnable grid (simplified: fixed grid for now to ensure stability)
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h) + grid_range[0]).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)
        
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (torch.rand(self.grid_size + self.spline_order, self.in_features, self.out_features) - 1/2) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_((self.scale_spline if self.scale_spline is not None else 1.0) * self.curve2coeff(self.grid.T[self.spline_order : -self.spline_order], noise))

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        
        grid = self.grid  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        
        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1] + \
                    (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:]
        
        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.
        """
        # Simplified initialization logic
        return torch.rand(self.out_features, self.in_features, self.grid_size + self.spline_order) * 0.1

    def forward(self, x):
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # Spline output
        # x shape: (batch, in_features)
        # bases shape: (batch, in_features, grid_size + spline_order)
        bases = self.b_splines(x)
        
        # spline_weight shape: (out_features, in_features, grid_size + spline_order)
        # We need to compute sum(bases * spline_weight) over the last dim, then sum over in_features
        
        # Efficient computation using einsum
        spline_output = torch.einsum("bij,oij->bo", bases, self.spline_weight)
        
        return base_output + spline_output

class KAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers_hidden=[64]):
        super(KAN, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(KANLayer(input_dim, layers_hidden[0]))
        
        # Hidden layers
        for i in range(len(layers_hidden) - 1):
            self.layers.append(KANLayer(layers_hidden[i], layers_hidden[i+1]))
            
        # Output layer
        self.layers.append(KANLayer(layers_hidden[-1], output_dim))
        
    def forward(self, x):
        # Flatten input if it's a sequence (batch, seq_len, features) -> (batch, seq_len * features)
        # Or simpler: just take the last time step for now, or average pooling
        # For this demo, let's flatten
        batch_size = x.size(0)
        x = x.view(batch_size, -1) 
        
        for layer in self.layers:
            x = layer(x)
        return x

class BiLSTM_Attention(nn.Module):
    """
    Bi-directional LSTM with Attention mechanism.
    Better suited for high-volatility/complex time series than simple MLPs or KANs.
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=2, dropout=0.2):
        super(BiLSTM_Attention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bi-directional LSTM
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, 
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # Attention Layer
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Output Layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # LSTM output: (batch, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)
        
        # Attention weights
        attn_weights = F.softmax(self.attention(lstm_out), dim=1) # (batch, seq_len, 1)
        
        # Context vector (weighted sum of LSTM outputs)
        context = torch.sum(attn_weights * lstm_out, dim=1) # (batch, hidden_dim * 2)
        
        # Final prediction
        out = self.fc(context)
        return out
