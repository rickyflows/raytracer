#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from utils import *

np.set_printoptions(precision=2, suppress=True)

def ray_color(ray: Ray):
    color_arr = np.empty_like(ray.direction)
    # (m, n, 3) / (m, n)
    unit_directions = ray.direction / np.linalg.norm(ray.direction, axis=2, keepdims=True)
    # lerp factor (m, n)
    a = 0.5 * (unit_directions[:, :, 1:2] + 1.0)
    color_arr = (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0)

    logical_hit_sphere = hit_sphere(vec(0, 0, -1), 0.5, ray)
    color_arr[logical_hit_sphere==True, :] = color(1,0,0)
    return color_arr

def hit_sphere(center: np.ndarray, radius: float, ray: Ray):
    oc = ray.origin - center
    b = 2 * np.tensordot(oc, ray.direction, axes=(0,2))
    # print('b', b)
    a = np.sum(np.copy(ray.direction**2), axis=2)
    c = np.dot(oc, oc) - radius * radius
    discriminant = b**2 - 4 * a * c
    # print(discriminant)
    return (discriminant >= 0)

def main():
    # Image
    aspect_ratio = 16 / 9
    image_width = 400
    image_height = int(image_width / aspect_ratio)
    image_channels = 3

    # Camera
    focal_length = 1.0
    viewport_height = 2.0
    viewport_width = viewport_height * image_width / image_height
    camera_center = vec(0.0,0.0,0.0)

    # Calculate the vectors across the horizontal and down the vertical viewport edges.
    viewport_u = vec(viewport_width, 0.0, 0.0)
    viewport_v = vec(0.0, -viewport_height, 0.0)

    # Calculate the horizontal and vertical delta vectors from pixel to pixel.
    pixel_delta_u = viewport_u / image_width
    pixel_delta_v = viewport_v / image_height

    # Calculate the location of the upper left pixel.
    viewport_upper_left = camera_center - vec(0.0, 0.0, focal_length) - viewport_u/2 - viewport_v/2
    pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)

    # Create image
    image = np.zeros((image_height, image_width, image_channels))
    image_indices = np.indices((image_height, image_width)).transpose(1, 2, 0)

    pixel_centers = pixel00_loc + pixel_delta_u * image_indices[:, :, 1:2] + pixel_delta_v * image_indices[:, :, 0:1]
    ray_directions = pixel_centers - camera_center
    ray = Ray(camera_center, ray_directions)
    image = ray_color(ray)


    plt.imshow(image)
    plt.savefig('renders/ray_with_sphere.png')
    plt.show()


if __name__=='__main__':
    main()
