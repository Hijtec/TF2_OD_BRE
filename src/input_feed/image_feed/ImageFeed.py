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

