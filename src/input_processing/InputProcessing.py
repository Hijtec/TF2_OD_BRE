"""A helper class providing methods for preprocessing inputs.
"""

from absl import logging

from src.utils.preprocessing_utils import is_image_blurry, equalize_image, crop_black_borders


class InputProcessing:
    def __init__(self, blurry_threshold=100):
        self.blurry_threshold = blurry_threshold
        self.equalized = False

    @staticmethod
    def equalize_image(image):
        return equalize_image(image)

    @staticmethod
    def crop_black_outlines(image):
        return crop_black_borders(image)

    def is_input_image_blurry(self, image):
        res = is_image_blurry(image, self.blurry_threshold)
        if res is True:
            logging.info("InputImage considered blurry")
            return True
        else:
            return False

