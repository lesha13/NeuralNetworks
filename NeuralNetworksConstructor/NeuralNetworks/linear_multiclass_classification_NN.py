from NeuralNetworks import NeuralNetwork
from Layers import DenseLayer
from activation_functions import softmax


class NNm(NeuralNetwork):
    def __init__(self):
        self.dense_layer1 = DenseLayer(2, 10)
        self.dense_layer2 = DenseLayer(10, 5)
        self.dense_layer3 = DenseLayer(5, 4, softmax)
