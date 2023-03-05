import numpy as np

from Layers import Layer
from decorators import timer


class NeuralNetwork(object):
    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, x: np.ndarray, y: np.ndarray, lr: float = 0.1) -> None:
        raise NotImplementedError

    @timer
    def train(self, data: np.ndarray, labels: np.ndarray, epochs: int = 100) -> None:
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            for x, y in zip(data, labels):
                self.backward(x, y)
        else:
            self.release_memory()

    def release_memory(self) -> None:
        for layer in self.__dict__.keys():
            getattr(self, layer).release_memory()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __enter__(self):
        Layer.save_memory = True
        return self

    def __exit__(self, exit_type, exit_value, exit_traceback):
        Layer.save_memory = False

    def __repr__(self):
        return (
                f"{self.__class__.__name__}(NeuralNetwork):\n" +
                "\n".join(f"{key}: {val}" for key, val in self.__dict__.items()) + "\n"
        )

    def __str__(self):
        return repr(self)
