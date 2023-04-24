import numpy as np

from Layers import Layer


class TrainableLayer(Layer):
    """
    Inherits base class Layer.
    Added attributes, methods needed for backpropagation
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
            activation function used in forward and backward propagation
        :_mean: int
            constant for initializing weights and bias
        :_deviation: int
            constant for initializing weights and bias
    -----
    Properties
        :weights: np.ndarray
            weights used for forward propagation. Only getter without setter
        :bias: np.ndarray
            bias used for forward propagation. Only getter without setter
    """
    _mean = 0
    _deviation = 0.1

    def __init__(self, *args, **kwargs):
        super(TrainableLayer, self).__init__()
        self._weights = None
        self._bias = None
        self.weighted_sum = None
        self.activation = None

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def bias(self) -> np.ndarray:
        return self._bias

    @property
    def output(self) -> np.ndarray:
        """
        Computes result of applying activation func (parameter derivative = False) on weighted sum
        -----
        :return: np.ndarray
            resulting numpy array
        """
        return self.activation(self.weighted_sum)

    @property
    def derivative_weighted_sum(self) -> np.ndarray:
        """
        Computes result of applying activation func (parameter derivative = True) on weighted sum
        -----
        :return: np.ndarray
            resulting numpy array
        """
        return self.activation(self.weighted_sum, True)

    def backward(self, *args, **kwargs) -> None:
        """
        Updates weights and bias
        -----
        :param args:
        :param kwargs:
        -----
        :return: None
        """
        raise NotImplementedError
