#!/usr/bin/env python3
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from typing import Union


def color(r: float, g: float, b: float) -> NDArray:
    return np.array((r, g, b))


def vec(x: float, y: float, z: float) -> NDArray:
    return np.array((x, y, z))


def unit_vector(arr: NDArray):
    return arr / np.linalg.norm(arr, axis=-1, keepdims=True)


class Ray:
    def __init__(self, origin: NDArray, direction: NDArray):
        self.origin = origin  # shape: (3)
        self.direction = direction  # shape: (m, n, 3)

    def at(self, t: Union[float, NDArray]):
        # t is either a scalar or (m, n, 1)
        return self.origin + np.multiply(t, self.direction)


@dataclass
class HitRecord:
    # If hits[i, j] is false then there no guarantees on other values at i, j
    hits: NDArray[np.bool_]  # shape: (m, n)
    times: NDArray[np.float64]  # shape: (m, n)
    points: NDArray[np.float64]  # shape: (m, n, 3)
    normals: NDArray[np.float64]  # shape: (m, n, 3)
    front_face: NDArray[np.bool_]  # shape: (m, n)

    def __init__(self, hits, times, points, ray, outward_normals):
        self.hits = hits
        self.times = times
        self.points = points
        self.set_face_normals(ray, outward_normals)

    def set_face_normals(self, ray: Ray, outward_normals: NDArray):
        # Sets the hit record normal vector.
        # NOTE: the parameter `outward_normal` is assumed to have unit length.
        self.front_face = np.einsum("ijk,ijk->ij", ray.direction, outward_normals) < 0
        self.normals = np.where(
            self.front_face[:, :, np.newaxis], outward_normals, -outward_normals
        )


class Hittable(ABC):
    @abstractmethod
    def hit(self, ray: Ray, tmin: float = 0, tmax: float = float("inf")) -> HitRecord:
        pass


class Sphere(Hittable):
    def __init__(self, center: NDArray, radius: float):
        self.center = center
        self.radius = radius

    def hit(self, ray: Ray, tmin: float = 0, tmax: float = float("inf")) -> HitRecord:
        oc = ray.origin - self.center
        a = np.sum(ray.direction**2, axis=-1)
        half_b = np.tensordot(oc, ray.direction, axes=(-1, -1))
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = half_b**2 - a * c

        # find the nearest root in the range [tmin, tmax]
        hits = discriminant >= 0
        sqrtd = np.sqrt(np.where(hits, discriminant, 0))
        smaller_roots = (-half_b - sqrtd) / a
        times = np.where(
            (smaller_roots <= tmin) | (smaller_roots >= tmax),
            (-half_b + sqrtd) / a,
            smaller_roots,
        )
        hits &= (times >= tmin) & (times <= tmax)
        points = ray.at(times[:, :, np.newaxis])
        outward_normals = (points - self.center) / self.radius

        return HitRecord(
            hits=hits,
            times=times,
            points=points,
            ray=ray,
            outward_normals=outward_normals,
        )
