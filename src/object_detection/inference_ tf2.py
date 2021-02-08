import tensorflow as tf
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
def infer_tf2(tensor_input, model_tf2):
    """Infers the tensor_input through TF2 model.
    Args:
        tensor_input: Tensor corresponding to the model input.
        model_tf2: SavedModel of TensorFlow2 type.
    Returns:
        detections_batch: Dictionary of detections.
    """
    print(f"Inferring through TF2 model of {model_tf2.__name__}")
    detections_batch = model_tf2(tensor_input)
    return detections_batch


@timer_wrapper
def infer_keras(tensor_input, model_keras):
    """Infers the tensor_input through TF2 model.
    Args:
        tensor_input: Tensor corresponding to the model input.
        model_keras: SavedModel of Keras type.
    Returns:
        predictions: List of detections / class score.
    """
    print(f"Inferring through Keras model of {model_keras.__name__}")
    predictions = model_keras.predict(tensor_input)
    return predictions
