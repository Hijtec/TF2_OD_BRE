"""A class providing a Detector capability using TF2 API.
This module contains a class needed for inference through TensorFlow2 SavedModel.
"""
import numpy as np
import tensorflow as tf
from absl import logging

from src.object_detection.inference.Detector import Detector
from src.utils.path_utils import check_path_existence
from src.utils.perf_utils import timer_wrapper


class DetectorTF2(Detector):
    def __init__(self, model_path):
        self.model = None

        self.model_path = check_path_existence(model_path, self.__class__.__name__)
        self.load_detector(self.model_path)

    @timer_wrapper
    def load_detector(self, inference_graph_tf2_dir):
        """
        :param inference_graph_tf2_dir: PATH to directory containing SavedModel and its weights.
        :return: Loaded TF2 SavedModel used for inference.
        """
        logging.info(f"Loading TF2 model from \n{inference_graph_tf2_dir}")
        model_tf2 = tf.saved_model.load(inference_graph_tf2_dir)
        logging.info("TF2 SavedModel load OK!")
        self.model = model_tf2

    @timer_wrapper
    def infer_tensor_input(self, tensor_input):
        """
        Infers a tensor through TF2 model.
        :param tensor_input: tf.Tensor input to neural network
        :return: dictionary of detections
        """
        if self.model is None:
            raise ValueError("Detection TF2 SavedModel has not been loaded.")
        logging.info(f"Detecting using TF2 model {self.model.__class__.__name__}")
        detection = self.model(tensor_input)
        return detection

    def infer_image(self, image, input_size):
        """
        Infers an image through DetectorTF2r.
        :param image: Numpy array containing input_image
        :param input_size: Tuple with contents of (input_height, input_width). If None, resizing skipped.
        :return: Dict of detection output
        """
        img_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.uint8)
        if input_size is not None:
            img_tensor = tf.image.resize_with_pad(img_tensor,
                                                  input_size[0],
                                                  input_size[1],
                                                  method=tf.image.ResizeMethod.BICUBIC)
        detection = self.infer_tensor_input(img_tensor)
        return detection

    @timer_wrapper
    def infer_images(self, images, input_size=None):
        """
        Infers images through DetectorTF2.
        :param input_size: Tuple with contents of (input_height, input_width).
        If None, no resizing with padding will be done.
        :param images: List of numpy arrays containing image data
        :return: List of Dict of classification output
        """
        detections = []
        for img in images:
            img_detection = self.infer_image(img, input_size)
            detections.append(img_detection)
        return detections
