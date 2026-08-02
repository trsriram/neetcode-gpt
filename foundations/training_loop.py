import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import torch

class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        N = X.shape[0]
        N_f = X.shape[1]
        X_ten = torch.from_numpy(X).to(torch.float32)
        Y_ten = torch.from_numpy(y).to(torch.float32).reshape(-1,1)
        w = torch.zeros((N_f, 1), dtype = torch.float32, requires_grad = True)
        b = torch.zeros(1, dtype = torch.float32, requires_grad = True)

        for eoch in range(epochs):
            y_hat = torch.matmul(X_ten, w) + b
            loss = torch.mean((y_hat - Y_ten) ** 2)
            loss.backward()
            with torch.no_grad():
                w -= lr * w.grad
                b -= lr * b.grad
                w.grad.zero_()
                b.grad.zero_()

        w_final = np.round(w.detach().numpy().flatten(), 5)
        b_final = round(b.detach().numpy().item(), 5)

        return (w_final, b_final)