class Loss(object):
    def __init__(self):
        self.loss = None

    def __repr__(self):
        return f"loss = {self.loss}"

    def __str__(self):
        return repr(self)
