import numpy as np
from numpy.random import normal

from Layers import Layer, TrainableLayer


class RadialBasisLayer(TrainableLayer):
    _mean = 0.5
    _deviation = 0.1

    def __init__(self, input_: int, output_: int):
        self._weights = normal(
            # TrainableLayer._mean,
            # TrainableLayer._deviation,
            RadialBasisLayer._mean,
            RadialBasisLayer._deviation,
            (output_, input_)
        )
        self._bias = normal(
            # TrainableLayer._mean,
            # TrainableLayer._deviation,
            RadialBasisLayer._mean,
            RadialBasisLayer._deviation,
            output_
        )
        self.weighted_sum = None
        self.activation = self._transform

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self._save_memory:
            self.weighted_sum = x
            return self.output
        else:
            return self._transform(x)

    def gradient(self, error, next_layer) -> np.ndarray:
        error = error @ next_layer.weights
        # error = (error / self._bias).T * self.derivative_weighted_sum
        return error

    def backward(self, error: np.ndarray, prev_layer, lr: float = 0.1) -> None:
        return self._transform(prev_layer, True)

    def release_memory(self) -> None:
        self.weighted_sum = None

    @staticmethod
    def gaussian(data, mu=0, sigma=1, derivative=False):
        diff = data - mu
        sum_of_squares = np.sum(diff ** 2)
        data = np.exp(
            -sum_of_squares / (2 * sigma ** 2)
        )
        if not derivative:
            return data
        else:
            return diff * data / sigma, sum_of_squares * data / (sigma ** 3)

    def _transform(self, x, derivative=False):
        if not derivative:
            return np.array([self.gaussian(x, mu, sigma, derivative) for mu, sigma in zip(self._weights, self._bias)])
        else:
            return [self.gaussian(x, mu, sigma, derivative) for mu, sigma in zip(self._weights, self._bias)]
