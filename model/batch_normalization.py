import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        X_np = np.array(x)
        X_mean = X_np.mean(axis=0)
        X_var = X_np.var(axis=0)
        running_mean_arr = np.array(running_mean)
        running_var_arr = np.array(running_var)
        if training:
            x_upd = (X_np - X_mean)/np.sqrt(X_var + eps)
            x_upd = x_upd* gamma  + beta
            running_mean_arr =  running_mean_arr * (1 - momentum) + X_mean * momentum
            running_var_arr = (1 - momentum) * running_var_arr + momentum * X_var
            return (x_upd.round(4).tolist(), running_mean_arr.round(4).tolist(), running_var_arr.round(4).tolist())
        
        x_upd = ((X_np - running_mean_arr)/np.sqrt(running_var_arr + eps))* gamma  + beta

        return (x_upd.round(4).tolist(), running_mean_arr.round(4).tolist(), running_var_arr.round(4).tolist())

