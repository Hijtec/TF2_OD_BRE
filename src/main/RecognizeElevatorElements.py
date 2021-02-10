"""A class wrapping the whole project.
TODO: Main class, describe what it does
"""
from src.elevator_controls_detection.ElementDetection import ElementDetection
from src.elevator_controls_detection.FloorButtonClassification import FloorButtonClassification
from src.input_feed.InputFeed import InputFeed

from absl import logging


class RecognizeElevatorElements:
    def __init__(self):
        self.input = InputFeed()
        self.detection = ElementDetection()
        self.classification = FloorButtonClassification()
        self.postprocessing = None  # TODO: these parts are not yet implemented
        self.visualization = None  # TODO: these parts are not yet implemented

    def recognize_elevator_elements(self):
        input_data = self.input.get_input_data_batch()
        if input_data is None:
            logging.warning('No InputData. Skipping.')
            return
        self.detection.detect_next_image(input_data['ImageData'])
        detection_data = self.detection.get_detection()
        # TODO: filter out only floor buttons from the detection_data
        return

    def loop(self):
        while True:
            self.recognize_elevator_elements()
