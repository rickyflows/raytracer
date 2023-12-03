#!/usr/bin/env python3
from utils import vec
from hittables import Sphere, HittableList
from camera import Camera


def main():
    # World
    world = HittableList()
    world.add(Sphere(vec(0, 0, -1), 0.5))
    world.add(Sphere(vec(0, -100.5, -1), 100))

    # Camera
    camera = Camera()
    camera.aspect_ratio = 16 / 9
    camera.image_width = 400

    camera.render(world)


if __name__ == "__main__":
    main()
