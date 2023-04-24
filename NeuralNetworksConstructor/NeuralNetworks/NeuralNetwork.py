import numpy as np
import matplotlib.pyplot as plt

from Layers import Layer, TrainableLayer
from decorators import timer
from loss_functions import Loss, CrossEntropyLoss


class NeuralNetwork(object):
    """
    Base class for all neural networks.
    Has methods for forward/backward propagation, training the model and releasing memory.
    Has context manager, which is used to save memory.
    """
    def __init__(self):
        raise NotImplementedError

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Does the forward propagation of the data.
        -----
        :param x: x: np.ndarray
            input data
        -----
        :return: np.ndarray
            resulting output (prediction)
        """
        layers = self._get_layers()
        for layer in layers:
            x = getattr(self, layer)(x)
        return x.squeeze()

    def backward(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 loss_function: Loss = CrossEntropyLoss,
                 lr: float | list[float] = 0.1,
                 ) -> None:
        """
        Does the forward propagation of the data.
        -----
        :param x: np.ndarray
            one training input data point
        :param y: np.ndarray
            one true target value
        :param loss_function: Loss
            loss function object
        :param lr: float
            "learning rate": some positive number usually between [e-1, e-6] which is used for smoothing the training
        -----
        :return: None
        """

        layers = self._get_layers(reverse=True)
        layers_count = len(self)

        if layers_count < 1:
            raise Exception

        flag = not isinstance(lr, float)
        if flag:
            lr = reversed(lr)

        def get_lr(learning_rate):
            try:
                return next(learning_rate) if flag else learning_rate
            except StopIteration:
                return 0.001

        layer = getattr(self, layers[0])
        loss = loss_function(self(x), y)
        error = layer.gradient(loss)

        if layers_count > 1:
            next_layer = getattr(self, layers[1])
            layer.backward(error, next_layer, get_lr(lr))

            for i in tuple(range(layers_count))[1:-1:]:
                layer = getattr(self, layers[i])
                prev_layer = getattr(self, layers[i+1])
                next_layer = getattr(self, layers[i-1])

                error = layer.gradient(error, next_layer)
                if isinstance(layer, TrainableLayer):
                    layer.backward(error, prev_layer, get_lr(lr))

            layer = getattr(self, layers[-1])
            next_layer = getattr(self, layers[-2])
            error = layer.gradient(error, next_layer)

        layer.backward(error, x, get_lr(lr))

    def release_memory(self) -> None:
        """
        Maps through all layers and calls release_memory() method on them.
        -----
        :return: None
        """
        layers = self._get_layers()
        for layer in layers:
            getattr(self, layer).release_memory()

    @timer
    def train(self,
              data: np.ndarray,
              labels: np.ndarray,
              loss_function: Loss = CrossEntropyLoss,
              lr: float = 0.1,
              epochs: int = 100,
              plot_loss: bool = False,
              ) -> None:
        """
        Trains the model by calling backward() method
        -----
        :param data: np.ndarray
            all training data
        :param labels: np.ndarray
            true target values
        :param loss_function: Loss
            loss function object
        :param epochs: int
            training epochs
        :param lr: float
            "learning rate": some positive number usually between [e-1, e-6] which is used for smoothing the training
        :param plot_loss: bool
            default: False
            if plot_loss = True plots the loss plot
        -----
        :return: None
        """

        epoch_loss = []
        for epoch in range(epochs):
            print(f"Epoch: {epoch}")
            for x, y in zip(data, labels):
                self.backward(x, y, loss_function, lr)

            loss = np.abs(np.array(loss_function.avg_loss)).mean()
            epoch_loss.append(loss)
            loss_function.drop_avg_loss()
            print(f"\tLoss: {loss}")
        else:
            self.release_memory()
            if plot_loss:
                plt.plot(epoch_loss, c="k")
                plt.title("Loss change through epochs")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.show()

    def _get_layers(self, reverse=False):
        if reverse:
            layers = tuple(reversed(self.__dict__.keys()))
        else:
            layers = tuple(self.__dict__.keys())
        return layers

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Calls forward() method.
        -----
        :param x: np.ndarray
        -----
        :return: np.ndarray
            numpy array of local gradients
        """
        return self.forward(x)

    def __enter__(self):
        """
        Used for changing to memory efficient mode.
        with {NeuralNetwork obj}:
            {some statement}
        -----
        :return: NeuralNetwork
        """
        Layer._save_memory = True
        return self

    def __len__(self):
        return len(self.__dict__)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        Layer._save_memory = False

    def __repr__(self):
        return (
                f"{self.__class__.__name__}(NeuralNetwork):\n" +
                "\n".join(f"{key}: {val}" for key, val in self.__dict__.items()) + "\n"
        )

    def __str__(self):
        return repr(self)
