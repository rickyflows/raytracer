#!/usr/bin/env python3
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
        self.origin = origin  # shape: (3)
        self.direction = direction  # shape: (m, n, 3)

    def at(self, t: Union[float, np.ndarray]):
        # t is either a scalar or (m, n, 1)
        return self.origin + np.multiply(t, self.direction)


@dataclass
class HitRecord:
    # If hits[i, j] is false then there no guarantees on other values at i, j
    hits: np.ndarray  # shape: (m, n)
    points: np.ndarray  # shape: (m, n, 3)
    normals: np.ndarray  # shape: (m, n, 3)
    times: np.ndarray  # shape: (m, n)


class Hittable(ABC):
    @abstractmethod
    def hit(self, ray: Ray, tmin: float = 0, tmax: float = float("inf")) -> HitRecord:
        pass


class Sphere(Hittable):
    def __init__(self, center: np.ndarray, radius: float):
        self.center = center
        self.radius = radius

    def hit(self, ray: Ray, tmin: float = 0, tmax: float = float("inf")) -> HitRecord:
        oc = ray.origin - self.center
        a = np.sum(ray.direction**2, axis=-1)
        half_b = np.tensordot(oc, ray.direction, axes=(-1, -1))
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = half_b**2 - a * c

        # find the nearest root in the range [tmin, tmax]
        hits = discriminant > 0
        sqrtd = np.sqrt(np.where(hits, 0, discriminant))
        smaller_roots = (-half_b - sqrtd) / a
        times = np.where(
            (smaller_roots <= tmin) | (smaller_roots >= tmax),
            (-half_b + sqrtd) / a,
            smaller_roots,
        )
        hits &= (times >= tmin) & (times <= tmax)
        points = ray.at(times[:, :, np.newaxis])
        normals = (points - self.center) / self.radius

        return HitRecord(hits=hits, points=points, normals=normals, times=times)
