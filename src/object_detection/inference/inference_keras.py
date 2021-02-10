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
    logging.info("Keras SavedModel load OK!")
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
    logging.info(f"Classifying using Keras model {model_keras.__name__}")
    classifications = model_keras.predict(tensor_input)
    return classifications


@timer_wrapper
def process_keras_classification(classifications, evenly_round_classification_scores=True):
    """Processes the TF2 detections batch
    Args:
        classifications: List of classifications / class score.
        evenly_round_classification_scores: Bool enabling the evenly rounded output of classification
    Returns:
        classifications: List of detections / class score.
    """
    logging.info(f"Processing Keras classification.")
    if evenly_round_classification_scores:
        classifications = np.around(classifications)
    return classifications


class ClassifierKeras(Classifier):
    def __init__(self, model_path):
        self.model = None
        self.tensor_input = None
        self.classifications = None
        self.output_raw_classifications = False

        self.model_path = check_path_existence(model_path, self.__name__)
        self.load_classifier(self.model_path)

    def load_classifier(self, detector_path):
        self.model = load_inference_graph_keras(detector_path)

    def infer_tensor_input(self, tensor_input):
        if self.model is None:
            raise ValueError("Classification Keras SavedModel has not been loaded.")

        self.tensor_input = tensor_input
        if self.output_raw_classifications is False:
            classifications = infer_keras_classification(tensor_input, self.model)
            self.classifications = process_keras_classification(classifications)
        else:
            self.classifications = infer_keras_classification(tensor_input, self.model)

    def get_classifier_output(self):
        return self.classifications

    def visualize_output(self):
        pass
        # TODO: visualization library
