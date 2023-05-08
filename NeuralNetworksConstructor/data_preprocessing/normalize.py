import numpy as np


def normalize(data: np.ndarray) -> np.ndarray:
    """
    Normalizes the data (all numbers will be in [0;1]).
    -----
    :key data: np.ndarray
    -----
    :return: np.ndarray
    """
    return (data - data.min()) / (data.max() - data.min())
