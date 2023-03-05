import numpy as np

from Layers import Layer


class MaxPoolLayer2d(Layer):
    def __init__(self):
        self.output = None
        self.indices = None

        super(MaxPoolLayer2d, self).__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        output, indices = [], []

        for point in x:
            o, i = self.max_pool_2d(point, True)
            output.append(o)
            indices.append(i)

        pooled, indices = np.array(output), np.array(indices)

        if not self.save_memory:
            self.output, self.indices = pooled, indices

        return pooled

    def gradient(self, error: np.ndarray, next_layer: Layer | None = None) -> np.ndarray:
        if isinstance(next_layer, Layer):
            error = next_layer.convolve(
                error,
                np.swapaxes(next_layer.weights, 0, 1)[::, ::, ::-1, ::-1],
                True,
            )
        self.indices[np.isnan(self.indices)] = error.reshape(-1)
        return self.indices

    def release_memory(self) -> None:
        self.output = None
        self.indices = None

    @staticmethod
    def max_pool_2d(data: np.ndarray, return_indices: bool = False) -> np.ndarray | tuple[np.ndarray]:
        # 2*2 max pool with a stride of 2
        ds0, ds1 = data.shape
        result = np.zeros([ds0 // 2, ds1 // 2])

        for j in range(2, ds0 + 1, 2):
            for i in range(2, ds1 + 1, 2):
                result[(j - 2) // 2, (i - 2) // 2] = data[j - 2:j, i - 2:i].max()

        if return_indices:
            indices = np.zeros_like(data, dtype="float32")
            for j in range(2, ds0 + 1, 2):
                for i in range(2, ds1 + 1, 2):
                    index = np.unravel_index(data[j - 2:j, i - 2:i].argmax(), data[j - 2:j, i - 2:i].shape)
                    indices[j - 2:j, i - 2:i][index[0], index[1]] = np.NaN
            return result, indices
        else:
            return result
