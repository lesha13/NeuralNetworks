import numpy as np


class CrossEntropyLoss(object):
    """
    Computes cross entropy (binary and multiclass) loss but with less computational cost.
    Can be used when output layer uses sigmoid for binary and softmax for multiclass classification
    """
    def __init__(self, prediction: np.ndarray, target: np.ndarray):
        self.loss = np.array(prediction - target).reshape([1, -1])

    def __repr__(self):
        return f"loss = {self.loss}"

    def __str__(self):
        return repr(self)
