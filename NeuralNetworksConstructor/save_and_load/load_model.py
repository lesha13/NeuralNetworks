import pickle

from NeuralNetworks import NeuralNetwork


def load_model(file: str) -> NeuralNetwork:
    """
    Function for loading saved model from .pickle file.
    -----
    :key file: str
        path to file
    -----
    :return: NeuralNetwork
        some subclass of NeuralNetwork
    """
    with open(file, "rb") as f:
        return pickle.load(f)
