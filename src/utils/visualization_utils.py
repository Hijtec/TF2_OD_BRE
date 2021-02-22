from PIL import Image
from matplotlib import pyplot as plt
import numpy as np


def __show_image_window(image):
    image.show()


def __show_plt_window(image):
    plt.imshow(image)
    plt.show()


def show_multiple_images_side_by_side(*images):
    resulting_image = []
    for img in images:
        np.vstack(resulting_image, img)
    __show_plt_window(resulting_image)
