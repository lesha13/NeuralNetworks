import numpy as np

from Layers import Layer
from decorators import timer


class NeuralNetwork(object):
    """
    Base class for all neural networks.
    Has methods for forward/backward propagation, training the model and releasing memory.
    Has context manager, which is used to save memory.
    """
    def __init__(self):
        raise NotImplementedError

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Does the forward propagation of the data.
        -----
        :param x: x: np.ndarray
            input data
        -----
        :return: np.ndarray
            resulting output (prediction)
        """
        # raise NotImplementedError
        for layer in self.__dict__.keys():
            x = getattr(self, layer)(x)
        return x.squeeze()

    def backward(self, x: np.ndarray, y: np.ndarray, lr: float = 0.1) -> None:
        """
        Does the forward propagation of the data.
        -----
        :param x: np.ndarray
            one training input data point
        :param y: np.ndarray
            one true target value
        :param lr: float
            "learning rate": some positive number usually between [e-1, e-6] which is used for smoothing the training
        -----
        :return: None
        """
        raise NotImplementedError

    @timer
    def train(self, data: np.ndarray, labels: np.ndarray, epochs: int = 100, lr: float = 0.1) -> None:
        """
        Trains the model by calling backward() method
        -----
        :param data: np.ndarray
            all training data
        :param labels: np.ndarray
            true target values
        :param epochs: int
            training epochs
        :param lr: float
            "learning rate": some positive number usually between [e-1, e-6] which is used for smoothing the training
        -----
        :return: None
        """
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            for x, y in zip(data, labels):
                self.backward(x, y, lr)
        else:
            self.release_memory()

    def release_memory(self) -> None:
        """
        Maps through all layers and calls release_memory() method on them.
        -----
        :return: None
        """
        for layer in self.__dict__.keys():
            getattr(self, layer).release_memory()

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

    def __enter__(self):
        """
        Used for changing to memory efficient mode.
        with {NeuralNetwork obj}:
            {some statement}
        -----
        :return: NeuralNetwork
        """
        Layer._save_memory = True
        return self

    def __exit__(self, exit_type, exit_value, exit_traceback):
        Layer._save_memory = False

    def __repr__(self):
        return (
                f"{self.__class__.__name__}(NeuralNetwork):\n" +
                "\n".join(f"{key}: {val}" for key, val in self.__dict__.items()) + "\n"
        )

    def __str__(self):
        return repr(self)
