from abc import ABC, abstractmethod


class LocationFeed(ABC):
    @abstractmethod
    def get_next_location(self):
        pass

    @abstractmethod
    def get_location(self):
        pass
