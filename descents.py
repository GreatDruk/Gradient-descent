import numpy as np
from enum import auto, Enum
from typing import Dict, Type


class LossFunctions(Enum):
    MSE = auto()
    MAE = auto()
    Huber = auto()
    LogCosh = auto()


class LearningRate:
    def __init__(self, lambd: float = 1e-3, s0: float = 1.0, p: float = 0.5):
        self.lambd = lambd
        self.s0 = s0
        self.p = p
        self.iteration: int = 0
    
    def __call__(self) -> float:
        self.iteration += 1
        learning_rate: float = self.lambd * (self.s0 / (self.s0 + self.iteration)) ** self.p
        return learning_rate


class BasicDescent:
    def __init__(self, dim: int, lambd: float = 1e-3, loss_function: LossFunctions = LossFunctions.MSE):
        self.w: np.ndarray = np.random.rand(dim)
        self.learning_rate: LearningRate = LearningRate(lambd)
        self.loss_function: LossFunctions = loss_function

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w
    
    def calculate_MSE(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y - y_pred) ** 2)
    
    def calculate_MAE(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y - y_pred))
    
    def calculate_Huber_loss(self, y: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
        def huber(diff: float) -> float:
            diff = np.abs(diff)
            if diff < delta:
                return diff ** 2 / 2
            else:
                return delta * (diff - delta / 2)

        huber_vec = np.vectorize(huber)
        return np.mean(huber_vec(y - y_pred))
    
    def calculate_Log_cosh(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.log(np.cosh(y - y_pred)))
    
    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred: np.ndarray = self.predict(x)
        if self.loss_function == LossFunctions.MSE:
            return self.calculate_MSE(y, y_pred)
        elif self.loss_function == LossFunctions.MAE:
            return self.calculate_MAE(y, y_pred)
        elif self.loss_function == LossFunctions.Huber:
            return self.calculate_Huber_loss(y, y_pred)
        elif self.loss_function == LossFunctions.LogCosh:
            return self.calculate_Log_cosh(y, y_pred)
        else:
            raise NotImplementedError(f"Unsupported loss: {self.loss_function}")
        
