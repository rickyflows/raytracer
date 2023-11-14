#!/usr/bin/env python3
import numpy as np

def color(r: float, g: float, b: float) -> np.ndarray:
    return np.array((r, g, b))

def vec(x: float, y: float, z: float) -> np.ndarray:
    return np.array((x, y, z))

class Ray:
    def __init__(self, origin: np.ndarray, direction: np.ndarray):
        self.origin = origin # shape: (3)
        self.direction = direction # shape: (m, n, 3)

    def at(self, t: float):
        return self.origin + t * self.direction
