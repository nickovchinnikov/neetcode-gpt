import numpy as np
from numpy.typing import NDArray
from typing import Tuple


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
        
        # init
        n = len(y)
        n_samples, n_features = X.shape
        W, b = np.zeros(n_features), 0
        y_hat = np.zeros_like(y)

        # Training loop
        for _ in range(epochs):
            y_hat = X @ W  + b

            L = (1/n) * np.sum((y_hat-y)**2)

            dLdw = (2/n) * (X.T @ (y_hat-y))
            dLdb = (2/n) * np.sum(y_hat-y)

            W -= lr * dLdw
            b -= lr * dLdb

        return (np.round(W, 5), round(b, 5))


