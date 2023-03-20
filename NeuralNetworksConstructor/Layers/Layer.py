import numpy as np


class Layer(object):
    """
    Base class for all neural network layers.
    Has methods for forward propagation, computing local gradient and releasing memory.
    -----
    Attributes
        :_save_memory: bool
            if _save_memory = False then layer does store the data needed for backpropagation
            else doesn't store the data needed for backpropagation
        :_info: str
            string used in __str__ and __repr__ to show the info about layer
    """
    _save_memory = False

    def __init__(self, *args, **kwargs):
        self._info = ""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Does the forward propagation of the data
        -----
        :param x: np.ndarray
        -----
        :return: np.ndarray
            numpy array of neuron outputs
        """
        raise NotImplementedError

    def gradient(self, *args, **kwargs) -> np.ndarray:
        """
        Computes the local gradient needed for backpropagation.
        ------
        :param args:
        :param kwargs:
        -------
        :return: np.ndarray
            numpy array of local gradients
        """
        raise NotImplementedError

    def release_memory(self) -> None:
        """
        Releases memory used for model training
        -----
        :return: None
        """
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Calls forward() method.
        -----
        :param x: np.ndarray
        -----
        :return: np.ndarray
            numpy array of local gradients
        """
        return self.forward(x)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._info})"

    def __str__(self):
        return repr(self)
