import numpy as np

from NeuralNetworks import NeuralNetwork
from Layers import ConvolutionLayer2d, MaxPoolLayer2d, FlattenLayer, LinearLayer

from loss_functions import CrossEntropyLoss
from activation_functions import softmax


class CNNm(NeuralNetwork):
    def __init__(self):
        super(CNNm, self).__init__()
        self.conv_layer1 = ConvolutionLayer2d(1, 10, 28, 28)  # 1*28*28 -> 10*26*26
        self.max_pool_layer1 = MaxPoolLayer2d()  # 10*26*26 -> 10*13*13

        self.flatten_layer = FlattenLayer()  # need to flatten: 10*13*13 -> ...

        self.linear_layer1 = LinearLayer(1690, 512)  # 1690 -> 256
        self.linear_layer2 = LinearLayer(512, 10, softmax)  # 256 -> 10

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.conv_layer1(x)
        x = self.max_pool_layer1(x)

        x = self.flatten_layer(x)

        x = self.linear_layer1(x)
        x = self.linear_layer2(x)

        return x.squeeze()

    def backward(self, x: np.ndarray, y: np.ndarray, lr: float = 0.1):
        loss = CrossEntropyLoss(self(x), y)

        error = self.linear_layer2.gradient(loss)
        self.linear_layer2.backward(error, self.linear_layer1, lr)

        error = self.linear_layer1.gradient(error, self.linear_layer2)
        self.linear_layer1.backward(error, self.flatten_layer, lr)

        error = self.flatten_layer.gradient(error, self.linear_layer1)

        error = self.max_pool_layer1.gradient(error)
        error = self.conv_layer1.gradient(error)
        self.conv_layer1.backward(error, x, lr)
