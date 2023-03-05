import numpy as np

from NeuralNetworks import NeuralNetwork
from Layers import LinearLayer
from loss_functions import CrossEntropyLoss
from activation_functions import softmax


class NNm(NeuralNetwork):
    def __init__(self):
        super(NNm, self).__init__()
        self.linear_layer1 = LinearLayer(2, 10)
        self.linear_layer2 = LinearLayer(10, 5)
        self.linear_layer3 = LinearLayer(5, 4, softmax)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.linear_layer1(x)
        x = self.linear_layer2(x)
        x = self.linear_layer3(x)
        return x.squeeze()

    def backward(self, x: np.ndarray, y: np.ndarray, lr: float = 0.1) -> None:
        loss = CrossEntropyLoss(self(x), y)

        error = self.linear_layer3.gradient(loss)
        self.linear_layer3.backward(error, self.linear_layer2, lr)

        error = self.linear_layer2.gradient(error, self.linear_layer3)
        self.linear_layer2.backward(error, self.linear_layer1, lr)

        error = self.linear_layer1.gradient(error, self.linear_layer2)
        self.linear_layer1.backward(error, x, lr)
