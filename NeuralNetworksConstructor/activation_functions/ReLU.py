import numpy as np


def ReLU(data: np.ndarray | None, derivative: bool = False) -> np.ndarray | None:
    """
    Applies ReLU function on numpy array.
    -----
    :param data: np.ndarray | None
        some input numpy array or None
    :param derivative: bool
        default: False
        if derivative = True computes derivative of ReLU on this data
    -----
    :return: np.ndarray | None
        resulting numpy array or None
    """
    if data is None:
        return None
    if not derivative:
        data[data < 0] = 0
    else:
        data[data < 0] = 0
        data[data > 0] = 1
    return data
