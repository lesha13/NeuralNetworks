import numpy as np
from numpy.random import normal

from Layers import Layer
from Layers import TrainableLayer
from activation_functions import ReLU


class ConvolutionLayer2d(TrainableLayer):
    """
    Inherits TrainableLayer.
    Implements forward(), gradient(), backward(), release_memory() methods, convolution operations.
    -----
    Attributes
        :_weights: np.ndarray
            weights used for forward propagation
            !don't call this attribute directly outside of this project, instead use property
        :_bias: np.ndarray
            bias used for forward propagation
            !don't call this attribute directly outside of this project, instead use property
        :weighted_sum: np.ndarray
            used for storing the result of forward propagation before applying the activation function
        :activation: callable
            activation function used in backward propagation
    """

    def __init__(self, input_: int, output_: int, l: int, w: int, kernel_size: int = 3, activation: callable = ReLU):
        """
        :param input_: int
            number of input data channels
        :param output_: int
            number of channels you want for output data to be
        :param l: int
            length of image
        :param w: int
            width of image
        :param kernel_size: int
            default kernel_size = 3
            size of a 2 dimensional filters (kernel_size, kernel_size)
        :param: callable
            activation function used in forward and backward propagation
        """
        # super(ConvolutionLayer2d, self).__init__()
        self._weights = normal(
            TrainableLayer._mean,
            TrainableLayer._deviation,
            (output_, input_, kernel_size, kernel_size)
        )
        self._bias = normal(
            TrainableLayer._mean,
            TrainableLayer._deviation,
            (output_, l - kernel_size + 1, w - kernel_size + 1)
        )
        self.weighted_sum = None
        self.activation = activation
        self._info = f"\n\tinput channels: {input_}, output channels: {output_}, " \
                     f"\n\tlength: {l}, width: {w}, kernel size: {kernel_size}, " \
                     f"\n\tactivation function: {activation.__name__}\n"

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Does the forward propagation of the data using convolution operations.
        -----
        :param x: np.ndarray
        -----
        :return: np.ndarray
        """
        x = x[np.newaxis, ...] if x.ndim == 2 else x
        result = self.convolve(x, self._weights) + self._bias

        if not self.save_memory:
            self.weighted_sum = result
            return self.output
        else:
            return self.activation(result)

    def gradient(self, error: np.ndarray) -> np.ndarray:
        """
        Computes the local gradient needed for backpropagation.
        -----
        :param error: np.ndarray
        -----
        :return: np.ndarray
        """
        return error @ self.output

    def backward(self, error: np.ndarray, prev_layer: Layer | np.ndarray, lr: float = 0.1) -> None:
        """

        -----
        :param error: np.ndarray
        :param prev_layer: Layer | np.ndarray
        :param lr: float
        -----
        :return: None
        """
        if isinstance(prev_layer, Layer):
            weights_update = np.array([
                [self.convolution_2d(output, e) for output in prev_layer.output]
                for e in error
            ])
        else:
            weights_update = np.array([
                [
                    self.convolution_2d(output, e) for output in
                    (prev_layer[np.newaxis, ...] if prev_layer.ndim == 2 else prev_layer)
                ]
                for e in error
            ])
        self._weights -= lr * weights_update
        self._bias -= lr * error

    def release_memory(self) -> None:
        """
        Sets weighted_sum to None
        -----
        :return: None
        """
        self.weighted_sum = None

    @staticmethod
    def convolution_2d(data: np.ndarray, kernel: np.ndarray, padding: bool = False) -> np.ndarray:
        """
        Static method used for 2d convolution between one 2 dimensional numpy array and one 2 dimensional numpy array
        -----
        :param data: np.ndarray
            2 dimensional numpy array (image or some other data)
        :param kernel: np.ndarray
            2 dimensional numpy array (kernel, filter)
        :param padding: bool
            default is False
            if True adds zeros so the resulting numpy arrays shape equals input data shape
        -----
        :return: np.ndarray
            convoluted 2 dimensional numpy array
        """
        ks0, ks1 = kernel.shape
        ps0, ps1 = data.shape

        if padding:
            zeros = np.zeros([ps0, ks1 // 2 * 2])
            data = np.hstack([zeros, data, zeros])
            ps0, ps1 = data.shape

            zeros = np.zeros([ks0 // 2 * 2, ps1])
            data = np.vstack([zeros, data, zeros])
            ps0, ps1 = data.shape

        result = np.zeros([ps0 - ks0 + 1, ps1 - ks1 + 1])

        for i in range(ks0, ps0 + 1):
            for j in range(ks1, ps1 + 1):
                i_, j_ = i - ks0, j - ks1
                result[i_, j_] = (data[i_:i, j_:j] * kernel).sum()

        return result

    @staticmethod
    def convolve(data: np.ndarray, kernels: np.ndarray, padding: bool = False) -> np.ndarray:
        """
        Static method used for 2d convolution between many 2 dimensional numpy array and many 2 dimensional numpy array
        -----
        :param data: np.ndarray
            2 dimensional numpy array (image or some other data)
        :param kernels: np.ndarray
            2 dimensional numpy array (kernel, filter)
        :param padding: bool
            default is False
            if True adds zeros so the resulting numpy arrays shape equals input data shape
        -----
        :return: np.ndarray
            numpy array of convoluted 2 dimensional numpy array
        """
        result = np.array([
            sum(ConvolutionLayer2d.convolution_2d(x, kernel, padding) for x, kernel in zip(data, kernel))
            for kernel in kernels
        ])
        return result
