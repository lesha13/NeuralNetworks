class Loss(object):
    _avg_loss = []

    def __init__(self):
        self.loss = None
        raise NotImplementedError

    @classmethod
    @property
    def avg_loss(cls):
        return cls._avg_loss

    @classmethod
    def drop_avg_loss(cls):
        cls._avg_loss = []

    def __repr__(self):
        return f"loss = {self.loss}"

    def __str__(self):
        return repr(self)
