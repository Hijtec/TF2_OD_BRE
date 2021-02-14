"""Abstract class Classifier.
This module contains a wrapper class around neural network inference.
"""
from abc import ABC, abstractmethod


class Classifier(ABC):
    @abstractmethod
    def load_classifier(self, detector_path):
        pass

    @abstractmethod
    def infer_tensor_input(self, tensor_input):
        pass

    @abstractmethod
    def infer_images(self, images, input_size):
        pass

    @abstractmethod
    def visualize_output(self):
        pass

    @abstractmethod
    def get_classifier_output(self):
        pass
