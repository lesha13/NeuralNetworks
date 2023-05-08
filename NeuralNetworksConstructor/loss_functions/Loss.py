import numpy as np


class Loss(object):
    _avg_loss = []

    def __init__(self):
        self.loss = None
        raise NotImplementedError

    @classmethod
    def avg_loss(cls):
        result = np.array(cls._avg_loss).mean()
        cls._avg_loss = []
        return result

    def __repr__(self):
        return f"Loss: {self.loss}"

    def __str__(self):
        return repr(self)
