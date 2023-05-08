import numpy as np


def cross_entropy_loss_binary(prediction: np.ndarray, target: np.ndarray, derivative: bool = True) -> np.ndarray:
    """
    Function for loss computing (Derivative of cross entropy loss).
    -----
    :key prediction: np.ndarray
        numpy array of model predictions
    :key target: np.ndarray
        numpy array of target values (labels)
    -----
    :return: np.ndarray
        resulting numpy array of loss
    """
    if derivative:
        result = ((1 - target) / (1 - prediction)) - (target / prediction)
    else:
        pass
    return result
