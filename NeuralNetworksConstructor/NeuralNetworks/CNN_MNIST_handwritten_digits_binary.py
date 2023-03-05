import numpy as np

from NeuralNetworks import NeuralNetwork
from Layers import ConvolutionLayer2d, MaxPoolLayer2d, FlattenLayer, LinearLayer

from loss_functions import CrossEntropyLoss


class CNNb(NeuralNetwork):
    def __init__(self):
        super(CNNb, self).__init__()
        self.conv_layer1 = ConvolutionLayer2d(1, 5, 28, 28)  # 1*28*28 -> 5*26*26
        self.max_pool_layer1 = MaxPoolLayer2d()  # 5*26*26 -> 5*13*13

        self.conv_layer2 = ConvolutionLayer2d(5, 20, 13, 13)  # 5*13*13 -> 20*11*11
        self.max_pool_layer2 = MaxPoolLayer2d()  # 20*11*11 -> 20*5*5

        self.flatten_layer = FlattenLayer()  # need to flatten: 20*5*5 -> 500

        self.linear_layer1 = LinearLayer(500, 250)  # 500 -> 250
        self.linear_layer2 = LinearLayer(250, 1)  # 250 -> 1

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.conv_layer1(x)
        x = self.max_pool_layer1(x)

        x = self.conv_layer2(x)
        x = self.max_pool_layer2(x)

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

        error = self.max_pool_layer2.gradient(error)
        error = self.conv_layer2.gradient(error)
        self.conv_layer2.backward(error, self.max_pool_layer1, lr)

        error = self.max_pool_layer1.gradient(error, self.conv_layer2)
        error = self.conv_layer1.gradient(error)
        self.conv_layer1.backward(error, x, lr)
