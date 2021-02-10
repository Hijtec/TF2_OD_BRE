"""A class handling the Floor Button Classification.
This module contains a Class capable of classifying floor buttons from their image.
"""
import tensorflow as tf
import numpy as np
from absl import logging
from src.main.flags_global import FLAGS
from src.object_detection.inference import inference_keras


class FloorButtonClassification:
    def __init__(self):
        self.input_image_button = None
        self.input_tensor_button = None
        self.output_classification = None
        self.Classifier = None

        self.__assign_classifier(FLAGS.classification_floor_button_model_path)
        logging.info('FloorButtonClassification created, ready for inference.')

    def __assign_classifier(self, model_path):
        if FLAGS.classification_floor_button_model_type == 'keras':
            self.Classifier = inference_keras.ClassifierKeras(model_path)
        else:
            # TODO: more types support
            raise ValueError('Other detectors not supported, use tf2 type.')

    def classify_next_image(self, image_data):
        """Runs classification on the given image_data.
        :param image_data: Numpy-like array with shape (width, height, n_channels)
        """
        image_tensor = tf.convert_to_tensor(np.expand_dims(image_data, 0), dtype=tf.float32)
        self.classify_next_tensor(image_tensor)

    def classify_next_tensor(self, image_tensor):
        """Runs classification on the given image_tensor.
        :param image_tensor: tf.tensor of the same shape as input layer of the Detector neural network.
        """
        self.Classifier.infer_tensor_input(image_tensor)
        self.output_classification = self.Classifier.get_detector_output()
        logging.info('FloorButtonClassification inference completed.')

    def get_classification(self):
        """Returns an output_classification field of ElementDetection.
        :return: Dictionary containing the detection output
        """
        if self.output_classification is None:
            raise ValueError('FloorButtonClassification output is None.')
        return self.output_classification
