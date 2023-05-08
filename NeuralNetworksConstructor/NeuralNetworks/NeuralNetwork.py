import numpy as np
import matplotlib.pyplot as plt

from Layers import Layer, TrainableLayer
from decorators import timer
from loss_functions import Loss, MeanSquareLoss


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
        :key x: x: np.ndarray
            input data
        -----
        :return: np.ndarray
            resulting output (prediction)
        """
        for layer in self:
            x = layer(x)

        return x.squeeze()

    def backward(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 loss_function: type(Loss) = MeanSquareLoss,
                 lr: float | list[float] = 0.1,
                 ) -> None:
        """
        Does the forward propagation of the data.
        -----
        :key x: np.ndarray
            one training input data point
        :key y: np.ndarray
            one true target value
        :key loss_function: Loss
            loss function object
        :key lr: float
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
                # return tuple(learning_rate)[0] if flag else learning_rate

        layer = layers[0]
        loss = loss_function(self(x), y)
        error = layer.gradient(loss)

        if layers_count > 1:
            prev_layer = layers[1]
            layer.backward(error, prev_layer, get_lr(lr))

            for i in tuple(range(layers_count))[1:-1:]:
                layer = layers[i]
                prev_layer = layers[i+1]
                next_layer = layers[i-1]

                error = layer.gradient(error, next_layer)
                if isinstance(layer, TrainableLayer):
                    layer.backward(error, prev_layer, get_lr(lr))

            layer = layers[-1]
            next_layer = layers[-2]
            error = layer.gradient(error, next_layer)

        layer.backward(error, x, get_lr(lr))

    @timer
    def train(self,
              data: np.ndarray,
              labels: np.ndarray,
              loss_function: type(Loss) = MeanSquareLoss,
              lr: float = 0.1,
              epochs: int = 100,
              plot_loss: bool = False,
              ) -> None:
        """
        Trains the model by calling backward() method
        -----
        :key data: np.ndarray
            all training data
        :key labels: np.ndarray
            true target values
        :key loss_function: Loss
            loss function object
        :key epochs: int
            training epochs
        :key lr: float
            "learning rate": some positive number usually between [e-1, e-6] which is used for smoothing the training
        :key plot_loss: bool
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

            loss = loss_function.avg_loss()
            print(f"\tLoss: {loss}")

        else:
            self.release_memory()
            if plot_loss:
                plt.plot(epoch_loss, c="k")
                plt.title("Loss change through epochs")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.show()

    def release_memory(self) -> None:
        """
        Maps through all layers and calls release_memory() method on them.
        -----
        :return: None
        """
        for layer in self:
            layer.release_memory()

    def _get_layers(self, reverse: bool = False) -> tuple[Layer]:
        if reverse:
            layers = tuple(reversed(self))
        else:
            layers = tuple(self.__dict__.values())

        return layers

    def _get_names(self, reverse: bool = False) -> tuple[Layer]:
        if reverse:
            layers = tuple(reversed(self.__dict__.keys()))
        else:
            layers = tuple(self.__dict__.keys())

        return layers

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Calls forward() method.
        -----
        :key x: np.ndarray
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

    def __exit__(self, exit_type, exit_value, exit_traceback):
        Layer._save_memory = False

    def __iter__(self):
        self.__layers = iter(self._get_layers())
        return self

    def __next__(self) -> Layer:
        try:
            return next(self.__layers)
        except StopIteration:
            del self.__layers
            raise StopIteration

    def __reversed__(self) -> iter:
        return reversed(self.__dict__.values())

    def __getitem__(self, key: int | slice | str | list[str] | tuple[str] | list[int] | tuple[int] | None = None):
        try:
            layers = self._get_layers()

            if key is None:
                return layers[-1]

            elif isinstance(key, (slice, int)):
                return layers[key]

            elif isinstance(key, str):
                return getattr(self, key)

            elif isinstance(key, (list, tuple)):

                if all(isinstance(i, int) for i in key):
                    return tuple(layers[i] for i in key)

                elif all(isinstance(n, str) for n in key):
                    return tuple(getattr(self, n) for n in key)

                else:
                    raise TypeError("All parameters must be same type (only str or int)")

            else:
                raise TypeError("Wrong key type")

        except IndexError:
            raise IndexError("No layer with such key")

        except AttributeError:
            raise AttributeError("No such attribute")

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(NeuralNetwork):\n"
            +
            (
                "\n".join(f"{key}: {val}" for key, val in self.__dict__.items())
                if self.__dict__
                else
                "\t*cricket sounds*"
            )
            +
            "\n"
        )

    def __str__(self):
        return repr(self)
