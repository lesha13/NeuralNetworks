import numpy as np

from loss_functions import Loss, squared_loss


class CrossEntropyLoss(Loss):
    """
    Computes cross entropy (binary and multiclass) loss but with less computational cost
    (All the extra computation shortened for result to be just a subtraction of prediction and target).
    Can be used when output layer uses sigmoid for binary and softmax for multiclass classification.
    """
    def __init__(self, prediction: np.ndarray, target: np.ndarray):
        # the reshape is needed if target is a single value
        self.loss = squared_loss(prediction, target).reshape([1, -1])
        # self._avg_loss = []
