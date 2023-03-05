import numpy as np


def squared_loss(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Function for loss computing (Derivative of squared loss).
    -----
    :param prediction: np.ndarray
        numpy array of model predictions
    :param target: np.ndarray
        numpy array of target values (labels)
    -----
    :return: np.ndarray
        resulting numpy array of loss
    """
    return prediction - target
