"""A class providing a Detector capability using Keras API.
This module contains a class needed for inference through Keras SavedModel.
"""
import numpy as np
import tensorflow as tf
from absl import logging

from src.object_detection.inference.Classifier import Classifier
from src.utils.path_utils import check_path_existence
from src.utils.perf_utils import timer_wrapper


@timer_wrapper
def load_inference_graph_keras(inference_graph_keras_dir):
    """Loads the inference graph.
    Args:
        inference_graph_keras_dir: Path to the Keras inference graph with embedded weights.
    Returns:
        model_keras: Loaded model used for inference.
    """
    logging.info(f"Loading Keras model from \n{inference_graph_keras_dir}")
    model_keras = tf.keras.models.load_model(inference_graph_keras_dir)
    latest_checkpoint = tf.train.latest_checkpoint(inference_graph_keras_dir)
    model_keras.load_weights(latest_checkpoint)
    logging.info("Keras SavedModel load OK! Weights loaded.")
    return model_keras


@timer_wrapper
def infer_keras_classification(tensor_input, model_keras):
    """Infers the tensor_input through TF2 model.
    Args:
        tensor_input: Tensor corresponding to the model input.
        model_keras: SavedModel of Keras type.
    Returns:
        classifications: List of classifications / class score.
    """
    logging.info(f"Classifying using Keras model {model_keras.__class__.__name__}")
    return model_keras.predict(tensor_input)


class ClassifierKeras(Classifier):
    def __init__(self, model_path):
        self.model = None
        self.tensor_input = None
        self.classifications = None

        self.model_path = check_path_existence(model_path, self.__class__.__name__)
        self.load_classifier(self.model_path)

    def load_classifier(self, detector_path):
        self.model = load_inference_graph_keras(detector_path)

    def infer_tensor_input(self, tensor_input):
        if self.model is None:
            raise ValueError("Classification Keras SavedModel has not been loaded.")
        return infer_keras_classification(tensor_input, self.model)

    def infer_image(self, image, input_size):
        """
        Infers an image through KerasClassifier.
        :param image: Numpy array containing input_image
        :param input_size: Tuple with contents of (input_height, input_width)
        :return: Dict of classification output
        """
        img_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
        img_tensor_resized = tf.image.resize_with_pad(img_tensor, input_size[0], input_size[1], method=tf.image.ResizeMethod.BICUBIC)
        return self.infer_tensor_input(img_tensor_resized)

    def infer_images(self, images, input_size):
        """
        Infers images through KerasClassifier.
        :param input_size: Tuple with contents of (input_height, input_width)
        :param images: List of numpy arrays containing image data
        :return: List of Dict of classification output
        """
        self.classifications = []
        for img in images:
            img_classification = self.infer_image(img, input_size)
            self.classifications.append(img_classification)
        return self.classifications

    def get_classifier_output(self):
        return self.classifications

    def visualize_output(self):
        pass
        # TODO: visualization library
