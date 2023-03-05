import numpy as np

from Layers import Layer


class FlattenLayer(Layer):
    def __init__(self):
        self.output = None

        super(FlattenLayer, self).__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        result = x.flatten()
        if not self.save_memory:
            self.output = result
        return result

    def gradient(self, error: np.ndarray, next_layer: Layer) -> np.ndarray:
        error = error @ next_layer.weights
        return error

    def release_memory(self) -> None:
        self.output = None
