import numpy as np


def zero_center(data: np.ndarray) -> np.ndarray:
    """
    Centers the data (the mean value of data will be equal to 0).
    -----
    :param data: np.ndarray
    -----
    :return: np.ndarray
    """
    return data - data.mean()
