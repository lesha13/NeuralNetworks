import numpy as np


def signum(data: np.ndarray | None, derivative: bool = False) -> np.ndarray | None:
    """
    Applies signum function on numpy array.
    -----
    :key data: np.ndarray | None
        some input numpy array or None
    :key derivative: bool, optional
        default: False
        if derivative = True computes derivative of signum on this data
    -----
    :return: np.ndarray | None
        resulting numpy array or None
    """
    if data is None:
        return None

    if not derivative:
        return np.where(data < 0, -1, 1)
    else:
        return np.zeros_like(data)
