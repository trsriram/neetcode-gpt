import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        std_ = math.sqrt(2.0/(fan_in + fan_out))
        torch.manual_seed(0)
        mat = torch.randn(fan_out, fan_in) * std_
        return mat.round(decimals=4).tolist()
        

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        std_ = math.sqrt(2.0/fan_in)
        torch.manual_seed(0)
        mat = torch.randn(fan_out, fan_in) * std_
        return mat.round(decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Reset seed ONCE at the start of generation sequence
        torch.manual_seed(0)
        
        dims = [input_dim] + [hidden_dim] * num_layers
        weights = []
        
        # Step 1: Generate all layer weight tensors sequentially
        for i in range(num_layers):
            fan_in = dims[i]
            fan_out = dims[i + 1]
            
            if init_type == "kaiming":
                std_ = math.sqrt(2.0 / fan_in)
            elif init_type == "xavier":
                std_ = math.sqrt(2.0 / (fan_in + fan_out))
            else:
                std_ = 1.0
                
            w = torch.randn(fan_out, fan_in) * std_
            weights.append(w)
            
        # Step 2: Generate input tensor x
        x = torch.randn(1, input_dim)
        
        # Step 3: Forward pass and record standard deviations
        stds = []
        for w in weights:
            x = x @ w.T
            x = torch.relu(x)
            stds.append(round(x.std().item(), 2))
            
        return stds

