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

        x, W1, b1, W2, b2, y_true = (
            np.array(x),
            np.array(W1),
            np.array(b1),
            np.array(W2),
            np.array(b2),
            np.array(y_true),
        )

        # Forward
        # Layer1
        z1 = W1 @ x + b1
        # Relu
        a1 = np.maximum(0, z1)

        # Layer2
        z2 = W2 @ a1 + b2

        n = len(z2)
        L = (1/n) * np.sum((z2 - y_true)**2)

        # Backward
        dL_z2 = (2 * (z2-y_true)) / n
        dL_dW2 = np.outer(dL_z2, a1)
        dL_db2 = dL_z2
        dL_da1 = dL_z2 @ W2
        dL_dz1 = np.where(z1 > 0, dL_da1, 0)
        dL_dW1 = np.outer(dL_dz1, x)
        dL_db1 = dL_dz1

        return {
            'loss': round(L, 5),
            'dW1': np.round(dL_dW1, 5),
            'db1': np.round(dL_db1, 5),
            'dW2': np.round(dL_dW2, 5),
            'db2': np.round(dL_db2, 5)
        }





