import numpy as np
import matplotlib.pyplot as plt
import time
from utils import Ray, color, unit_vector, vec
from hittables import HittableList


class Camera:
    def __init__(self):
        self.aspect_ratio: float = 1.0  # ratio of image width to height
        self.image_width: int = 100  # Rendered image width pixel count

    def render(self, world: HittableList):
        self.__initialize()
        # Create image
        image = np.zeros((self._image_height, self.image_width, self.image_channels))
        image_indices = np.indices((self._image_height, self.image_width)).transpose(
            1, 2, 0
        )

        pixel_centers = (
            self._pixel00_loc
            + self._pixel_delta_u * image_indices[:, :, 1:2]
            + self._pixel_delta_v * image_indices[:, :, 0:1]
        )
        ray_directions = pixel_centers - self._camera_center
        ray = Ray(self._camera_center, ray_directions)
        print("Starting image render...")
        start_time = time.time()
        image = self.__ray_color(ray, world)
        print("Time to render image:", time.time() - start_time)

        plt.imshow(image)
        plt.axis("off")
        plt.savefig("renders/world.png", bbox_inches="tight")
        plt.show()

    def __initialize(self):
        self._image_height = int(self.image_width / self.aspect_ratio)
        self.image_channels = 3

        # Camera
        focal_length = 1.0
        viewport_height = 2.0
        viewport_width = viewport_height * self.image_width / self._image_height
        self._camera_center = vec(0.0, 0.0, 0.0)

        # Calculate the vectors across the horizontal and down the vertical viewport edges.
        viewport_u = vec(viewport_width, 0.0, 0.0)
        viewport_v = vec(0.0, -viewport_height, 0.0)

        # Calculate the horizontal and vertical delta vectors from pixel to pixel.
        self._pixel_delta_u = viewport_u / self.image_width
        self._pixel_delta_v = viewport_v / self._image_height

        # Calculate the location of the upper left pixel.
        viewport_upper_left = (
            self._camera_center
            - vec(0.0, 0.0, focal_length)
            - viewport_u / 2
            - viewport_v / 2
        )
        self._pixel00_loc = viewport_upper_left + 0.5 * (
            self._pixel_delta_u + self._pixel_delta_v
        )

    def __ray_color(self, ray: Ray, world: HittableList):
        color_arr = np.empty_like(ray.direction)
        # (m, n, 3) / (m, n)
        unit_directions = unit_vector(ray.direction)
        # lerp factor (m, n)
        a = 0.5 * (unit_directions[:, :, 1:2] + 1.0)
        color_arr = (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0)

        # draw sphere
        hit_record = world.hit(ray)
        color_arr[hit_record.hits, :] = (0.5 * (1 + hit_record.normals))[
            hit_record.hits, :
        ]
        return color_arr
