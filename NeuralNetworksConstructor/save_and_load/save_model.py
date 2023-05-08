import pickle

from NeuralNetworks import NeuralNetwork


def save_model(file: str, model: NeuralNetwork) -> None:
    """
    Function for saving model to .pickle file.
    -----
    :key file: str
        path to file
    :key model: NeuralNetwork
        some subclass of NeuralNetwork
    -----
    :return: None
    """
    with open(file, "wb") as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
