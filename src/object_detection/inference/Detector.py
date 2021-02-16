"""Abstract class Detector.
This module contains a wrapper class around neural network inference.
"""
from abc import ABC, abstractmethod


class Detector(ABC):
    @abstractmethod
    def load_detector(self, detector_path):
        pass

    @abstractmethod
    def infer_tensor_input(self, tensor_input):
        pass

    @abstractmethod
    def infer_images(self, images, input_size=None):
        pass
