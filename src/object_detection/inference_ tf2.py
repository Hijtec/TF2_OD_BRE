import tensorflow as tf
import numpy as np
from src.utils.perf_utils import timer_wrapper


@timer_wrapper
def load_inference_graph_tf2(inference_graph_tf2_dir):
    """Loads the inference graph.
    Args:
        inference_graph_tf2_dir: Path to the TF2 inference graph with embedded weights.
    Returns:
        model_tf2: Loaded model used for inference.
    """
    print(f"Loading TF2 model from \n{inference_graph_tf2_dir}")
    model_tf2 = tf.saved_model.load(inference_graph_tf2_dir)
    print("Load OK!")
    return model_tf2


@timer_wrapper
def load_inference_graph_keras(inference_graph_keras_dir):
    """Loads the inference graph.
    Args:
        inference_graph_keras_dir: Path to the Keras inference graph with embedded weights.
    Returns:
        model_keras: Loaded model used for inference.
    """
    print(f"Loading Keras model from \n{inference_graph_keras_dir}")
    model_keras = tf.keras.models.load_model(inference_graph_keras_dir)
    print("Load OK!")
    return model_keras


@timer_wrapper
def infer_tf2_detection(tensor_input, model_tf2):
    """Infers the tensor_input through TF2 model.
    Args:
        tensor_input: Tensor corresponding to the model input.
        model_tf2: SavedModel of TensorFlow2 type.
    Returns:
        detections_batch: Dictionary of detections.
    """
    print(f"Detecting using TF2 model {model_tf2.__name__}")
    detections_batch = model_tf2(tensor_input)
    return detections_batch


@timer_wrapper
def infer_keras_classification(tensor_input, model_keras):
    """Infers the tensor_input through TF2 model.
    Args:
        tensor_input: Tensor corresponding to the model input.
        model_keras: SavedModel of Keras type.
    Returns:
        classifications: List of classifications / class score.
    """
    print(f"Classifying using Keras model {model_keras.__name__}")
    classifications = model_keras.predict(tensor_input)
    return classifications


@timer_wrapper
def process_tf2_detection(detections_batch, convert_classes_to_int=True):
    """Processes the TF2 detections batch
    Args:
        detections_batch: Dictionary of detections.
        convert_classes_to_int: Bool converting the classes to int64(float by default)
    Returns:
        predictions: List of detections / class score.
    """
    num_detections = int(detections_batch.peek('num_detections'))
    print(f"Processing {num_detections} detections.")
    detections = {k: v[0, :num_detections].numpy() for k, v in detections_batch.items()}
    if convert_classes_to_int:
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)


@timer_wrapper
def process_keras_classification(classifications, evenly_round_classification_scores=True):
    """Processes the TF2 detections batch
    Args:
        classifications: List of classifications / class score.
        evenly_round_classification_scores: Bool enabling the evenly rounded output of classification
    Returns:
        classifications: List of detections / class score.
    """
    print(f"Processing classification.")
    if evenly_round_classification_scores:
        classifications = np.around(classifications)
    return classifications
