#!/usr/bin/env python3
import numpy as np
from typing import Union

def color(r: float, g: float, b: float) -> np.ndarray:
    return np.array((r, g, b))

def vec(x: float, y: float, z: float) -> np.ndarray:
    return np.array((x, y, z))

def unit_vector(arr: np.ndarray):
    return arr / np.linalg.norm(arr, axis=-1, keepdims=True)

class Ray:
    def __init__(self, origin: np.ndarray, direction: np.ndarray):
        self.origin = origin # shape: (3)
        self.direction = direction # shape: (m, n, 3)

    def at(self, t: Union[float, np.ndarray]):
        # t is either a scalar or (m, n, 1)
        return self.origin + np.multiply(t, self.direction)
