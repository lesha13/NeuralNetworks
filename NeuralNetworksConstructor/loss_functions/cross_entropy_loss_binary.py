import numpy as np


def cross_entropy_loss_binary(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Function for loss computing (Derivative of cross entropy loss).
    -----
    :param prediction: np.ndarray
        numpy array of model predictions
    :param target: np.ndarray
        numpy array of target values (labels)
    -----
    :return: np.ndarray
        resulting numpy array of loss
    """
    return ((1 - target) / (1 - prediction)) - (target / prediction)
