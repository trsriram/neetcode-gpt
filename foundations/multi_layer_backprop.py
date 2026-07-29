import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        y_pred = 1/(1 + np.exp(-(np.sum(x*w) + b)))
        error = y_true - y_pred
        weight_grad = -np.round(error * y_pred * (1-y_pred) * x, 5)
        bias_grad = -np.round(error * y_pred * (1-y_pred), 5)
        return (weight_grad, bias_grad)
