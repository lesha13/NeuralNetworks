import numpy as np


def hyperbolic_tangent(data: np.ndarray | None, derivative: bool = False) -> np.ndarray | None:
    """
    Applies hyperbolic tangent function on numpy array.
    -----
    :key data: np.ndarray | None
        some input numpy array or None
    :key derivative: bool, optional
        default: False
        if derivative = True computes derivative of hyperbolic tangent on this data
    -----
    :return: np.ndarray | None
        resulting numpy array or None
    """
    if data is None:
        return None

    data = (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))
    if not derivative:
        return data
    else:
        return 1 - data ** 2
