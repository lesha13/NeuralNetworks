import numpy as np

from Layers import Layer, ConvolutionLayer2d


class MaxPoolLayer2d(Layer):
    def __init__(self):
        super(MaxPoolLayer2d, self).__init__()
        self.output = None
        self.indices = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self._save_memory:
            output = []
            indices = []

            for point in x:
                o, i = self.max_pool_2d_1ch(point, True)
                output.append(o)
                indices.append(i)

            pooled, indices = np.array(output), np.array(indices)
            self.output, self.indices = pooled, indices
        else:
            pooled = np.array([self.max_pool_2d_1ch(point) for point in x])

        return pooled

    def gradient(self, error: np.ndarray, next_layer: Layer) -> np.ndarray:
        if isinstance(next_layer, ConvolutionLayer2d):
            error = ConvolutionLayer2d.convolve(
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
    def max_pool_2d_1ch(data: np.ndarray, return_indices: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
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

    @staticmethod
    def max_pool_2d_mch(data: np.ndarray, return_indices: bool = False) -> np.ndarray | tuple[np.ndarray]:
        return np.array([
            MaxPoolLayer2d.max_pool_2d_1ch(x, return_indices) for x in data
        ])
