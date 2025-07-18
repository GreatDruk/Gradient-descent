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
        diff: np.ndarray = np.abs(y - y_pred)
        grad: np.ndarray = np.where(diff < delta, diff ** 2 / 2, delta * (diff - delta / 2))
        return np.mean(grad)
    
    def calculate_Log_cosh(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.log(np.cosh(y - y_pred)))
    
    def calculate_loss(self, x: np.ndarray, y: np.ndarray) -> float:
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
        

class GradientDescent(BasicDescent):
    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        w_diff = -self.learning_rate() * gradient
        self.w += w_diff
        return w_diff

    def calculate_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_pred = self.predict(x)

        if self.loss_function == LossFunctions.MSE:
            return (2 * x.T @ (y_pred - y)) / len(y)
        elif self.loss_function == LossFunctions.MAE:
            return (x.T @ np.sign(y_pred - y)) / len(y)
        elif self.loss_function == LossFunctions.Huber:
            diff: np.ndarray     = y_pred - y
            delta: float = 1.0
            grad: np.ndarray = np.where(np.abs(diff) < delta, diff, delta * np.sign(diff))
            return (x.T @ grad) / len(y)
        elif self.loss_function == LossFunctions.LogCosh:
            return (x.T @ np.tanh(y_pred - y)) / len(y)
        else:
            raise NotImplementedError(f"Unsupported loss: {self.loss_function}")

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        gradient: np.ndarray = self.calculate_gradient(x, y)
        return self.update_weights(gradient)
    

class StochasticDescent(GradientDescent):
    def __init__(self, dim: int, lambd: float = 1e-3, batch: int = 10, loss_function: LossFunctions = LossFunctions.MSE):
        super().__init__(dim, lambd, loss_function)
        self.batch = batch
    
    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        batch_size = min(self.batch, len(y))
        ind: np.ndarray = np.random.randint(0, len(y), batch_size)
        x = x[ind]
        y = y[ind]
        return super().step(x, y)


class MomentumDescent(GradientDescent):
    def __init__(self, dim: int, lambd: float = 1e-3, alp: float = 0.9, loss_function: LossFunctions = LossFunctions.MSE):
        super().__init__(dim, lambd, loss_function)
        self.alp: float = alp
        self.h: np.ndarray = np.zeros(dim)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        self.h = self.alp * self.h + self.learning_rate() * gradient
        self.w -= self.h
        return -self.h
    
    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        gradient: np.ndarray = super().calculate_gradient(x, y)
        return self.update_weights(gradient)


class Adam(GradientDescent):
    def __init__(self, dim: int, lambd: float = 1e-3, eps: float = 1e-8, beta_1: float = 0.9, beta_2: float = 0.999,
                  loss_function: LossFunctions = LossFunctions.MSE):
        super().__init__(dim, lambd, loss_function)
        self.eps: float = eps
        self.beta_1: float = beta_1
        self.beta_2: float = beta_2

        self.m: np.ndarray = np.zeros(dim)
        self.v: np.ndarray = np.zeros(dim)

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        self.iteration += 1

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * gradient ** 2

        m_: np.ndarray = self.m / (1 - self.beta_1 ** self.iteration)
        v_: np.ndarray = self.v / (1 - self.beta_2 ** self.iteration)

        w_diff = -self.learning_rate() * m_ / (np.sqrt(v_) + self.eps)
        self.w += w_diff

        return w_diff
    
    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        gradient: np.ndarray = super().calculate_gradient(x, y)
        return self.update_weights(gradient)


def start_descent(config: dict) -> BasicDescent:
    name: str = config.get('name', 'full')
    loss_name: str = config.get('loss', 'MSE')
    kwargs: dict = config.get('kwargs', {})
    descent_map: Dict[str, Type[BasicDescent]] = {
        'full': GradientDescent,
        'stochastic': StochasticDescent,
        'momentum': MomentumDescent,
        'adam': Adam,
    }

    try:
        loss_function = LossFunctions[loss_name]
    except KeyError:
        valid_losses = [e.name for e in LossFunctions]
        raise ValueError(f'Incorrect loss function, you can use one of these: {valid_losses}')

    if name not in descent_map:
        raise ValueError(f'Incorrect descent name, you can use one of these: {descent_map.keys()}')

    kwargs['loss_function'] = loss_function
    descent_class: Type[BasicDescent] = descent_map[name]

    return descent_class(**kwargs)
