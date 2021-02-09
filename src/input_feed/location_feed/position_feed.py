"""A script providing inputs to the system.
This module contains a Class handling the inputs into the system
"""
from src.main.flags_global import FLAGS
from src.input_feed.location_feed.LocationFeed import LocationFeed
# TODO: PLACEHOLDER


class PosConstFeedSync(LocationFeed):
    def __init__(self):
        self.location = 'elevator'

    def get_next_location(self):
        FLAGS.robot_position = self.location  # Always in elevator

    def get_location(self):
        self.get_next_location()
        return self.location
