import numpy as np
from PIL import Image
from src.utils.image_utils import equalize, crop_to_area_above_threshold
import cv2


def detect_blur(image, threshold=100):
    focus_measure = variance_of_laplacian(image)

    if focus_measure < threshold:
        return True
    else:
        return False


def variance_of_laplacian(image):
    """
    :param image: ndarray of shape (width, height, channels)
    :return: float variance of laplacian on that image
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()
    # compute the Laplacian of the image and then return the focus measure, which is simply the variance of the
    # Laplacian


def is_image_blurry(image, threshold=100):
    return detect_blur(image, threshold)


def equalize_image(image):
    return equalize(image)


def crop_black_borders(image):
    return crop_to_area_above_threshold(image, threshold=0)

