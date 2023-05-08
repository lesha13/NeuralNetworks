import string
import random

from NeuralNetworks import NeuralNetwork
from Layers import Layer


class SequentialNeuralNetwork(NeuralNetwork):
    def __init__(self, layers: Layer | list[Layer] | tuple[Layer] | dict[str, Layer] = None):
        if layers is not None:
            self.add(layers)

    def add(self, layers: Layer | list[Layer] | tuple[Layer] | dict[str, Layer], index: int = -1):
        if isinstance(index, int):

            names = self._get_names()

            if index >= 0:
                move = names[index::]

            else:
                length = len(names)
                move = names[length + index + 1::]

            add = {name: getattr(self, name) for name in move}

            if move:
                self.delete(move)

            if isinstance(layers, Layer):
                setattr(self, self.__generate_string(), layers)

            elif isinstance(layers, (list, tuple)):

                if all(isinstance(l, Layer) for l in layers):
                    for l in layers:
                        setattr(self, self.__generate_string(), l)

                else:
                    raise TypeError(f"Passed argument contains not only Layer type objects")

            elif isinstance(layers, dict):

                if (
                    all(isinstance(l, Layer) for l in layers.values())
                    and
                    all(isinstance(n, str) for n in layers.keys())
                ):
                    self.__check_names(layers.keys())
                    for n, l in layers.items():
                        setattr(self, n, l)

                else:
                    raise TypeError(f"Passed argument keys must be str and values must be Layer type")

            else:
                raise TypeError(f"Passed argument is not a Layer")

            if move:
                self.add(add)

            return self

        else:
            raise TypeError("Index must be int")

    def delete(self, index: int | slice | str | list[str] | tuple[str] | list[int] | tuple[int] | None = None):
        try:
            names = self._get_names()

            if index is None:
                delattr(self, names[-1])

            elif isinstance(index, int):
                delattr(self, names[index])

            elif isinstance(index, slice):
                for n in names[index]:
                    delattr(self, n)

            elif isinstance(index, str):
                delattr(self, index)

            elif isinstance(index, (list, tuple)):

                if all(isinstance(i, int) for i in index):
                    for n in (names[i] for i in index):
                        delattr(self, n)

                elif all(isinstance(n, str) for n in index):
                    for n in index:
                        delattr(self, n)

                else:
                    raise TypeError("All parameters must be same type (only str or int)")

            else:
                raise TypeError("Invalid key type (only str or int)")

            return self

        except IndexError:
            raise IndexError("No such layers(s) to delete")

    def __setitem__(self, key: Layer | list[Layer] | tuple[Layer] | dict[str, Layer], value: int = -1):
        self.add(value, key)

    def __generate_string(self, length: int = 10):
        try:
            result = "".join(random.choice(string.ascii_letters) for _ in range(length))
            self.__check_names(result)
            return result

        except AttributeError:
            return self.__generate_string(length)

    def __check_names(self, names: str | list[str]):
        attrs = set(self.__dict__.keys())
        names = {names} if isinstance(names, str) else set(names)

        if bool(attrs & names):
            raise AttributeError("The name(s) already exists")
