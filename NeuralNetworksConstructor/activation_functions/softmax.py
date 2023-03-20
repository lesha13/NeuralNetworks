import numpy as np


def softmax(data: np.ndarray | None, derivative: bool = False) -> np.ndarray | None:
    """
    Applies softmax function on numpy array.
    -----
    :param data: np.ndarray | None
        some input numpy array or None
    :param derivative: bool, optional
        default: False
        if derivative = True computes derivative of softmax on this data
    -----
    :return: np.ndarray | None
        resulting numpy array or None

    !!! Derivative not ready !!!
    """
    if data is None:
        return None

    if not derivative:
        e_x = np.exp(data - np.max(data))
        return e_x / e_x.sum()
    else:
        pass
