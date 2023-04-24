import numpy as np


def binary_step(data: np.ndarray | None, derivative: bool = False) -> np.ndarray | None:
    """
    Applies binary step function on numpy array.
    -----
    :param data: np.ndarray | None
        some input numpy array or None
    :param derivative: bool, optional
        default: False
        if derivative = True computes derivative of binary step on this data
    -----
    :return: np.ndarray | None
        resulting numpy array or None
    """
    if data is None:
        return None

    if not derivative:
        return np.where(data < 0, 0, 1)
    else:
        return np.zeros_like(data)
