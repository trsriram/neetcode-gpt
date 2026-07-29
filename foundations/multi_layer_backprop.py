import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                                x: List[float],
                                W1: List[List[float]], b1: List[float],
                                W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        x_arr = np.array(x)    # Input vector
        W1_arr = np.array(W1)  # Layer 1 weights
        b1_arr = np.array(b1)  # Layer 1 biases
        W2_arr = np.array(W2)  # Layer 2 weights
        b2_arr = np.array(b2)  # Layer 2 biases
        y_true_arr = np.array(y_true) # Output vector

        z1 = x_arr @ W1_arr.T + b1_arr
        a1 = np.maximum(z1, 0)
        z2 = a1 @ W2_arr.T + b2_arr
        loss = np.sum((y_true_arr - z2) ** 2)
        

        dZ2 = 2.0 * (z2 - y_true_arr)
        dW2_arr = np.outer(dZ2, a1)
        db2_arr = dZ2

        z1_grad = np.where(z1 > 0, 1.0, 0.0)
        da1 = dZ2 @ W2_arr
        dz1 = da1*z1_grad
        db1_arr = dz1
        dW1_arr = np.outer(dz1, x_arr)
        
        
        return {
            'loss': np.round(loss, 4),
            'dW1': np.round(dW1_arr, 4),
            'dW2': np.round(dW2_arr, 4),
            'db1': np.round(db1_arr, 4),
            'db2': np.round(db2_arr, 4),
        }
