#!/usr/bin/env python3
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from typing import Union, List


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


@dataclass
class HitRecord:
    def __init__(
        self,
        hits: NDArray[np.bool_],  # shape: (m, n)
        times: NDArray[np.float64],  # shape: (m, n)
        points: NDArray[np.float64],  # shape: (m, n, 3)
        normals: NDArray[np.float64],  # shape: (m, n, 3)
        front_face: NDArray[np.bool_],  # shape: (m, n)
    ):
        # If hits[i, j] is false then there no guarantees on other values at i, j
        self.hits = hits
        self.times = times
        self.points = points
        self.normals = normals
        self.front_face = front_face

    @classmethod
    def init_with_ray(cls, hits, times, points, ray, outward_normals):
        normals, front_face = cls.get_face_normals(ray, outward_normals)
        return cls(hits, times, points, normals, front_face)

    @staticmethod
    def get_face_normals(ray: Ray, outward_normals: NDArray):
        # Sets the hit record normal vector.
        # NOTE: the parameter `outward_normal` is assumed to have unit length.
        front_face = np.einsum("ijk,ijk->ij", ray.direction, outward_normals) < 0
        normals = np.where(
            front_face[:, :, np.newaxis], outward_normals, -outward_normals
        )
        return normals, front_face


class Hittable(ABC):
    @abstractmethod
    def hit(self, ray: Ray, ray_t: Interval = Interval()) -> HitRecord:
        pass


class HittableList(Hittable):
    def __init__(self, objects: List[Hittable] = []):
        self.objects = objects

    def add(self, object):
        self.objects.append(object)

    def hit(self, ray: Ray, ray_t: Interval = Interval()) -> HitRecord:
        # TODO: memory optimization and ray_t optimization
        N = len(self.objects)
        record_list = []
        for object in self.objects:
            record_list.append(object.hit(ray, ray_t))

        hits_stacked = np.stack([record.hits for record in record_list])
        masked_times_stacked = np.ma.stack(
            [
                np.ma.masked_array(record_list[i].times, mask=~hits_stacked[i])
                for i in range(N)
            ]
        )
        min_indices = np.ma.argmin(masked_times_stacked, axis=0)

        hits_final = np.any(hits_stacked, axis=0)
        times_final = np.ma.filled(masked_times_stacked.min(axis=0), fill_value=np.inf)
        m, n = hits_final.shape

        # Calculate remaining values
        points_list = [record.points for record in record_list]
        normals_list = [record.normals for record in record_list]
        front_face_list = [record.front_face for record in record_list]

        points_final = np.empty(shape=(m, n, 3), dtype=np.float64)
        normals_final = np.empty(shape=(m, n, 3), dtype=np.float64)
        front_face_final = np.empty(shape=(m, n), dtype=np.bool_)
        for i in range(N):
            mask = min_indices == i
            points_final[mask] = points_list[i][mask]
            normals_final[mask] = normals_list[i][mask]
            front_face_final[mask] = front_face_list[i][mask]

        return HitRecord(
            hits=hits_final,
            times=times_final,
            points=points_final,
            normals=normals_final,
            front_face=front_face_final,
        )


class Sphere(Hittable):
    def __init__(self, center: NDArray, radius: float):
        self.center = center
        self.radius = radius

    def hit(self, ray: Ray, ray_t: Interval = Interval()) -> HitRecord:
        oc = ray.origin - self.center
        a = np.sum(ray.direction**2, axis=-1)
        half_b = np.tensordot(oc, ray.direction, axes=(-1, -1))
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = half_b**2 - a * c

        # find the nearest root in the range [ray_t.mins, ray_t.maxs]
        hits = discriminant >= 0
        sqrtd = np.sqrt(np.where(hits, discriminant, 0))
        smaller_roots = (-half_b - sqrtd) / a
        times = np.where(
            ~ray_t.surrounds(smaller_roots),
            (-half_b + sqrtd) / a,
            smaller_roots,
        )
        hits &= ray_t.surrounds(times)
        points = ray.at(times[:, :, np.newaxis])
        outward_normals = (points - self.center) / self.radius

        return HitRecord.init_with_ray(
            hits=hits,
            times=times,
            points=points,
            ray=ray,
            outward_normals=outward_normals,
        )
