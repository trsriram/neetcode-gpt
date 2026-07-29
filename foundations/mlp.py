import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        num_layers = len(weights)
        cur = x
        for i in range(num_layers):
            z = cur @ weights[i] + biases[i]
            a = np.maximum(z, 0)
            cur = a
        return np.round(cur, 5)
