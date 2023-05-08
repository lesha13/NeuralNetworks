from NeuralNetworks import NeuralNetwork
from Layers import DenseLayer
from activation_functions import signum


class Perceptron(NeuralNetwork):
    def __init__(self):
        self.dense_layer = DenseLayer(2, 1, signum)
