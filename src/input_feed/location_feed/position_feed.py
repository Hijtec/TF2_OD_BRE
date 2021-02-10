"""A script providing inputs to the system.
This module contains a Class handling the inputs into the system
"""
from src.input_feed.location_feed.LocationFeed import LocationFeed
from src.main.flags_global import FLAGS
# TODO: PLACEHOLDER


class PosConstFeedSync(LocationFeed):
    def __init__(self):
        self.location = None

    def get_next_location(self):
        self.location = FLAGS.robot_position  # Always in location specified by FLAGS

    def get_location(self):
        self.get_next_location()
        return self.location
