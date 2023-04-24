import numpy as np
from numpy.random import normal

from Layers import Layer, TrainableLayer
from activation_functions import sigmoid
from loss_functions import Loss


class DenseLayer(TrainableLayer):
    """
    Inherits TrainableLayer.
    Implements forward(), gradient(), backward(), release_memory() methods.
    """
    def __init__(self, input_: int, output_: int, activation: callable = sigmoid):
        self._weights = normal(
            TrainableLayer._mean,
            TrainableLayer._deviation,
            (output_, input_)
        )
        self._bias = normal(
            TrainableLayer._mean,
            TrainableLayer._deviation,
            (output_, 1)
        )
        self.weighted_sum = None
        self.activation = activation

        self._info = f"\n\tinput size: {input_}, output size: {output_}, " \
                     f"activation function: {activation.__name__}\n"

    def forward(self, data: np.ndarray) -> np.ndarray:
        """
        Does the forward propagation of the data using liner transformations.
        -----
        :param data: np.ndarray
            numpy array of data that need to be transformed
        -----
        :return: np.ndarray
            resulting numpy array of transformed data
        """
        result = data @ self._weights.T + self._bias.T
        if not self._save_memory:
            self.weighted_sum = result

            return self.output
        else:

            return self.activation(result)

    def gradient(self, error: np.ndarray | Loss, next_layer: Layer | None = None) -> np.ndarray:
        """
        Computes the local gradient needed for backpropagation.
        -----
        :param error: np.ndarray
            loss or local gradients that needs to be propagated
        :param next_layer: next_layer: Layer | None
        """
        if isinstance(error, Loss):

            return error.loss
        else:
            if isinstance(next_layer, Layer):
                error = error @ next_layer.weights * self.derivative_weighted_sum
            else:
                error = error * self.derivative_weighted_sum

            return error

    def backward(self, error: np.ndarray, prev_layer: Layer | np.ndarray, lr: float = 0.1) -> None:
        """
        Updates weights and bias
        -----
        :param error: np.ndarray
            numpy array of calculated local gradients
            if this is the last layer, there are no next_layer so pass None
            else pass next layer object
        :param prev_layer: Layer | np.ndarray
            if this is the first layer, there is no prev_layer so pass input data
            else pass previous layer object
        :param lr: float
            value usually between 10^-1 and 10^-6.
            Makes the changes to weights and bias smaller, so the training becomes smoother.
        -----
        :return: None
        """
        error = lr * error.T

        if isinstance(prev_layer, Layer):
            prev_layer = prev_layer.output

        self._weights -= error * prev_layer
        self._bias -= error

    def release_memory(self) -> None:
        """
        Sets weighted_sum to None, so it doesn't take memory
        -----
        :return: None
        """
        self.weighted_sum = None
