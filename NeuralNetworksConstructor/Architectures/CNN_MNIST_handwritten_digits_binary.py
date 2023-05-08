from NeuralNetworks import NeuralNetwork
from Layers import Conv2dLayer, MaxPool2dLayer, FlattenLayer, DenseLayer


class CNNb(NeuralNetwork):
    def __init__(self):
        self.conv_layer1 = Conv2dLayer(1, 5, 28, 28)  # 1*28*28 -> 5*26*26
        self.max_pool_layer1 = MaxPool2dLayer()  # 5*26*26 -> 5*13*13

        self.conv_layer2 = Conv2dLayer(5, 20, 13, 13)  # 5*13*13 -> 20*11*11
        self.max_pool_layer2 = MaxPool2dLayer()  # 20*11*11 -> 20*5*5

        self.flatten_layer = FlattenLayer()  # need to flatten: 20*5*5 -> 500

        self.dense_layer1 = DenseLayer(500, 250)  # 500 -> 250
        self.dense_layer2 = DenseLayer(250, 1)  # 250 -> 1
