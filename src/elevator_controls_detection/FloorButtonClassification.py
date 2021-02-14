"""A class handling the Floor Button Classification.
This module contains a Class capable of classifying floor buttons from their image.
"""
import tensorflow as tf
import numpy as np
from absl import logging
from src.object_detection.inference import ClassifierKeras
from src.main.flags_global import FLAGS


class FloorButtonClassification:
    def __init__(self):
        self.Classifier = None

        self.__assign_classifier(FLAGS.classification_floor_button_model_path)
        logging.info('FloorButtonClassification created, ready for inference.')

    def __assign_classifier(self, model_path):
        if FLAGS.classification_floor_button_model_type == 'keras':
            self.Classifier = ClassifierKeras.ClassifierKeras(model_path)
        else:
            # TODO: more types support
            raise ValueError('Other detectors not supported, use tf2 type.')

    def classify_next_images(self, image_data, input_size=(224, 224)):
        """Runs classification on the given image_data.
        :param input_size: Tuple with contents of (input_height, input_width)
        :param image_data: List of image numpy array data
        """
        output = self.Classifier.infer_images(image_data, input_size)
        logging.info('FloorButtonClassification inference completed.')
        return output

    def classify_next_tensor(self, image_tensor):
        """Runs classification on the given image_tensor.
        :param image_tensor: tf.tensor of the same shape as input layer of the Detector neural network.
        """
        classification = self.Classifier.infer_tensor_input(image_tensor)
        logging.info('FloorButtonClassification inference completed.')
        return classification
