from __future__ import annotations
import numpy as np
from typing import List
from descents import BasicDescent, start_descent


class LinearRegression:
    def __init__(self, config: dict, tolerance: float = 1e-4, max_iter: int = 500):
        self.descent: BasicDescent = start_descent(config)

        self.tolerance: float = tolerance
        self.max_iter: int = max_iter
        self.loss_history: List[float] = []

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.descent.predict(x)
    
    def calculate_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        return self.descent.calculate_loss(x, y)
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegression:
        self.loss_history = [self.calculate_loss(x, y)]
        for i in range(self.max_iter):
            new_weight: np.ndarray = self.descent.step(x, y)
            error = np.linalg.norm(new_weight)

            if np.isnan(new_weight).any():
                break
            
            self.loss_history.append(self.calculate_loss(x, y))

            if error < self.tolerance:
                break

        return self