"""A class providing a Detector capability using TF2 API.
This module contains a class needed for inference through TensorFlow2 SavedModel.
"""
import numpy as np
import tensorflow as tf
from absl import logging

from src.object_detection.inference.Detector import Detector
from src.utils.path_utils import check_path_existence
from src.utils.perf_utils import timer_wrapper


@timer_wrapper
def load_inference_graph_tf2(inference_graph_tf2_dir):
    """Loads the inference graph.
    Args:
        inference_graph_tf2_dir: Path to the TF2 inference graph with embedded weights.
    Returns:
        model_tf2: Loaded model used for inference.
    """
    logging.info(f"Loading TF2 model from \n{inference_graph_tf2_dir}")
    model_tf2 = tf.saved_model.load(inference_graph_tf2_dir)
    logging.info("TF2 SavedModel load OK!")
    return model_tf2


@timer_wrapper
def infer_tf2_detection(tensor_input, model_tf2):
    """Infers the tensor_input through TF2 model.
    Args:
        tensor_input: Tensor corresponding to the model input.
        model_tf2: SavedModel of TensorFlow2 type.
    Returns:
        detections_batch: Dictionary of detections.
    """
    logging.info(f"Detecting using TF2 model {str(model_tf2)}")
    detections_batch = model_tf2(tensor_input)
    return detections_batch


@timer_wrapper
def process_tf2_detection(detections_batch, convert_classes_to_int=True):
    """Processes the TF2 detections batch
    Args:
        detections_batch: Dictionary of detections.
        convert_classes_to_int: Bool converting the classes to int64(float by default)
    Returns:
        detections: Dictionary of detections, processed.
    """
    # TODO: not used atm
    num_detections = int(detections_batch.pop('num_detections'))
    logging.info(f"Processing {num_detections} TF2 detections.")
    detections = {k: v[0, :num_detections].numpy() for k, v in detections_batch.items()}
    if convert_classes_to_int:
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    return detections


class DetectorTF2(Detector):
    def __init__(self, model_path):
        self.model = None
        self.tensor_input = None
        self.detections = None
        self.output_raw_detections = False

        self.model_path = check_path_existence(model_path, self.__class__.__name__)
        self.load_detector(self.model_path)

    def load_detector(self, detector_path):
        self.model = load_inference_graph_tf2(detector_path)

    def infer_tensor_input(self, tensor_input):
        if self.model is None:
            raise ValueError("Detection TF2 SavedModel has not been loaded.")

        self.tensor_input = tensor_input
        self.detections = infer_tf2_detection(tensor_input, self.model)

    def get_detector_output(self):
        return self.detections

    def visualize_output(self):
        pass
        # TODO: visualization library
