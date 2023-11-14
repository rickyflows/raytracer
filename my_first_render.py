#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2, suppress=True)


def main():
    image_height = 256
    image_width = 256
    image_channels = 3
    img = np.empty((image_channels, image_height, image_width))
    img[0,:,:] = np.linspace(0, 1, image_width)
    img[1,:,:] = img[0,:,:].T
    img[2,:,:] = np.zeros([image_height, image_width])

    plt.imshow(img.transpose(1, 2, 0))
    plt.savefig('renders/my_first_render.png')
    plt.show()


if __name__=='__main__':
    main()
