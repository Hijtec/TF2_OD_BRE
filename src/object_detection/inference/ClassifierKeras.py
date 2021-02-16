"""A class providing a Detector capability using Keras API.
This module contains a class needed for inference through Keras SavedModel.
"""
import numpy as np
import tensorflow as tf
from absl import logging

from src.object_detection.inference.Classifier import Classifier
from src.utils.path_utils import check_path_existence
from src.utils.perf_utils import timer_wrapper


class ClassifierKeras(Classifier):
    def __init__(self, model_path):
        self.model = None
        self.tensor_input = None
        self.classifications = None

        self.model_path = check_path_existence(model_path, self.__class__.__name__)
        self.load_classifier(self.model_path)

    @timer_wrapper
    def load_classifier(self, inference_graph_keras_dir):
        """
        Loads an inference graph.
        :param inference_graph_keras_dir: PATH to Keras inference graph model directory, which includes weights
        :return: Keras model instance - Loaded model used for inference
        """
        logging.info(f"Loading Keras model from \n{inference_graph_keras_dir}")
        model_keras = tf.keras.models.load_model(inference_graph_keras_dir)
        latest_checkpoint = tf.train.latest_checkpoint(inference_graph_keras_dir)
        logging.info("Keras SavedModel load OK!")
        model_keras.load_weights(latest_checkpoint)
        logging.info("Keras SavedModel weights load OK!")
        self.model = model_keras

    @timer_wrapper
    def infer_tensor_input(self, tensor_input):
        """
        Infers a tensor through Keras model.
        :param tensor_input: tf.Tensor input to neural network
        :return: dictionary of classifications.
        """
        if self.model is None:
            raise ValueError("Classification Keras SavedModel has not been loaded.")
        logging.info(f"Classifying using Keras model {self.model.__class__.__name__}")
        classification = self.model.predict(tensor_input)
        return classification

    def infer_image(self, image, input_size):
        """
        Infers an image through KerasClassifier.
        :param image: Numpy array containing input_image
        :param input_size: Tuple with contents of (input_height, input_width). If None, resizing skipped.
        :return: Dict of classification output
        """
        img_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
        if input_size is not None:
            img_tensor = tf.image.resize_with_pad(img_tensor,
                                                  input_size[0],
                                                  input_size[1],
                                                  method=tf.image.ResizeMethod.BICUBIC)
        return self.infer_tensor_input(img_tensor)

    def infer_images(self, images, input_size=None):
        """
        Infers images through KerasClassifier.
        :param input_size: Tuple with contents of (input_height, input_width).
        If None, no resizing with padding will be done.
        :param images: List of numpy arrays containing image data
        :return: List of Dict of classification output
        """
        self.classifications = []
        for img in images:
            img_classification = self.infer_image(img, input_size)
            self.classifications.append(img_classification)
        return self.classifications
