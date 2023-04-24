import numpy as np

from Layers import Layer, Conv2dLayer


class MaxPool2dLayer(Layer):
    def __init__(self):
        super(MaxPool2dLayer, self).__init__()
        self.output = None
        self.indices = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self._save_memory:
            result, indices = MaxPool2dLayer.max_pool_2d_mch(x, True)
            self.output, self.indices = result, indices

        else:
            result = self.max_pool_2d_mch(x)

        return result

    def gradient(self, error: np.ndarray, next_layer: Layer) -> np.ndarray:
        if isinstance(next_layer, Conv2dLayer):
            error = Conv2dLayer.convolve(
                error,
                Conv2dLayer.rotate_kernel_180(
                    np.swapaxes(next_layer.weights, 0, 1)
                ),
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
    def max_pool_2d_mch(data: np.ndarray, return_indices: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if return_indices:
            result = []
            indices = []

            for point in data:
                o, i = MaxPool2dLayer.max_pool_2d_1ch(point, True)
                result.append(o)
                indices.append(i)

            result, indices = np.array(result), np.array(indices)

            return result, indices
        else:
            result = np.array([
                MaxPool2dLayer.max_pool_2d_1ch(x) for x in data
            ])

            return result
