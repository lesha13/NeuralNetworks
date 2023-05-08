from NeuralNetworks import NeuralNetwork
from Layers import Conv2dLayer, MaxPool2dLayer, FlattenLayer, DenseLayer
from activation_functions import softmax


class CNNm(NeuralNetwork):
    def __init__(self):
        self.conv_layer1 = Conv2dLayer(1, 10, 28, 28)  # 1*28*28 -> 10*26*26
        self.max_pool_layer1 = MaxPool2dLayer()  # 10*26*26 -> 10*13*13

        self.flatten_layer = FlattenLayer()  # need to flatten: 10*13*13 -> ...

        self.dense_layer1 = DenseLayer(1690, 512)  # 1690 -> 256
        self.dense_layer2 = DenseLayer(512, 10, softmax)  # 256 -> 10
