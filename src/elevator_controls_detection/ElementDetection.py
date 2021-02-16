"""A class handling the Element Detection.
This module contains a Class capable of detecting elevator elements in an image.
"""
import numpy as np
import tensorflow as tf
from absl import logging

from src.main.flags_global import FLAGS
from src.object_detection.inference import DetectorTF2


class ElementDetection:
    def __init__(self):
        self.input_image = None
        self.input_tensor = None
        self.output_detection = None
        self.Detector = None

        self.__assign_detector(FLAGS.detector_elements_model_path)
        logging.info('ElementDetection created, ready for inference.')

    def __assign_detector(self, model_path):
        if FLAGS.detection_elements_model_type == 'tf2':
            self.Detector = DetectorTF2.DetectorTF2(model_path)
        else:
            # TODO: more types support
            raise ValueError('Other detectors not supported, use tf2 type.')

    def detect_next_image(self, image_data, input_shape=None):
        """Runs detection on the given image_data.
        :param image_data: Numpy-like array with shape (width, height, n_channels)
        :param input_shape: Tuple with contents of (input_height, input_width)
        """
        detections = self.Detector.infer_images(image_data, input_shape)
        logging.info('ElementDetection inference completed.')
        return detections

    def detect_next_tensor(self, image_tensor, input_shape=None):
        """Runs detection on the given image_tensor.
        :param image_tensor: tf.tensor of the same shape as input layer of the Detector neural network.
        :param input_shape: Tuple with contents of (input_height, input_width)
        """
        detection = self.Detector.infer_tensor_input(image_tensor, input_shape)
        logging.info('ElementDetection inference completed.')
        return detection
