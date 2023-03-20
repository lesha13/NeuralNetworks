import numpy as np


def sigmoid(data: np.ndarray | None, derivative: bool = False) -> np.ndarray | None:
    """
    Applies sigmoid function on numpy array.
    -----
    :param data: np.ndarray | None
        some input numpy array or None
    :param derivative: bool, optional
        default: False
        if derivative = True computes derivative of sigmoid on this data
    -----
    :return: np.ndarray | None
        resulting numpy array or None
    """
    if data is None:
        return None

    data = (1 / (1 + np.exp(-data)))
    if not derivative:
        return data
    else:
        return data * (1 - data)
