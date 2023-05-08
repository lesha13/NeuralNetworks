import numpy as np

from loss_functions import Loss


class CrossEntropyLoss(Loss):
    """
    Computes cross entropy (binary and multiclass) loss but with less computational cost
    (All the extra computation shortened for result to be just a subtraction of prediction and target).
    Can be used when output layers uses sigmoid for binary and softmax for multiclass classification.
    """

    def __init__(self, prediction: np.ndarray, target: np.ndarray):
        # the reshape is needed if target is a single value
        loss = (prediction - target).reshape([1, -1])
        self.loss = loss
        self._avg_loss.append(-(target * np.log(prediction)).sum())

    @staticmethod
    def cross_entropy_loss(prediction, t):
        pass
