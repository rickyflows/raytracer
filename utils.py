import numpy as np

from numpy.typing import NDArray
from typing import Union


def color(r: float, g: float, b: float) -> NDArray:
    return np.array((r, g, b))


def vec(x: float, y: float, z: float) -> NDArray:
    return np.array((x, y, z))


def unit_vector(arr: NDArray):
    return arr / np.linalg.norm(arr, axis=-1, keepdims=True)


class Interval:
    def __init__(
        self,
        mins: NDArray[np.float64] = np.array([0]),
        maxs: NDArray[np.float64] = np.array([np.inf]),
    ):
        assert (
            mins.shape == maxs.shape
        ), "Error Instantiating Interval: dimensions of mins and maxs disagree"
        self.mins = mins
        self.maxs = maxs

    def contains(self, x: NDArray[np.float64]):
        return (x >= self.mins) & (x <= self.maxs)

    def surrounds(self, x: NDArray[np.float64]):
        return (x > self.mins) & (x < self.maxs)


class Ray:
    def __init__(self, origin: NDArray, direction: NDArray):
        self.origin = origin  # shape: (3)
        self.direction = direction  # shape: (m, n, 3)

    def at(self, t: Union[float, NDArray]):
        # t is either a scalar or (m, n, 1)
        return self.origin + np.multiply(t, self.direction)
