"""Abstract Class ImageFeed.
This module contains a wrapper class around ImageInput providers.
"""
from abc import ABC, abstractmethod


class ImageFeed(ABC):
    @abstractmethod
    def get_next_frame(self):
        pass

    @abstractmethod
    def get_frame(self):
        pass

    @abstractmethod
    def show_frame(self):
        pass


class ImageFeedEmpty(ImageFeed):
    def get_next_frame(self):
        pass

    def get_frame(self):
        pass

    def show_frame(self):
        pass
