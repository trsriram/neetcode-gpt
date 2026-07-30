import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        x_arr = np.array(x)
        var = np.mean(x_arr ** 2)
        std = (var + eps) ** 0.5
        x_arr = x_arr / std * gamma
        return x_arr.round(4).tolist()
