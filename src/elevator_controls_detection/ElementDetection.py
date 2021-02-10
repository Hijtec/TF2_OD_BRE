"""A class handling the Element Detection.
This module contains a Class capable of detecting elevator elements in an image.
"""
import tensorflow as tf
import numpy as np
from absl import logging

from src.object_detection.inference import DetectorTF2
from src.main.flags_global import FLAGS


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

    def detect_next_image(self, image_data):
        """Runs detection on the given image_data.
        :param image_data: Numpy-like array with shape (width, height, n_channels)
        """
        image_tensor = tf.convert_to_tensor(np.expand_dims(image_data, 0), dtype=tf.uint8)
        self.detect_next_tensor(image_tensor)

    def detect_next_tensor(self, image_tensor):
        """Runs detection on the given image_tensor.
        :param image_tensor: tf.tensor of the same shape as input layer of the Detector neural network.
        """
        self.Detector.infer_tensor_input(image_tensor)
        self.output_detection = self.Detector.get_detector_output()
        logging.info('ElementDetection inference completed.')

    def get_detection(self):
        """Returns an output_detection field of ElementDetection.
        :return: Dictionary containing the detection output
        """
        if self.output_detection is None:
            raise ValueError('The ElementDetection output is None.')
        return self.output_detection
