"""A class wrapping the whole project.
TODO: Main class, describe what it does
"""
from src.elevator_controls_detection.ElementDetection import ElementDetection
from src.elevator_controls_detection.FloorButtonClassification import FloorButtonClassification
from src.input_feed.InputFeed import InputFeed
from src.main.flags_global import FLAGS
from absl import logging

from src.utils.detection_utils import non_max_suppress_detections, filter_one_category_from_detections_nms, create_category_index


class RecognizeElevatorElements:
    def __init__(self):
        self.input = InputFeed()
        self.detection = ElementDetection()
        # TODO: disabled for time saving self.classification = FloorButtonClassification()
        self.postprocessing = None  # TODO: these parts are not yet implemented
        self.visualization = None  # TODO: these parts are not yet implemented

    def recognize_elevator_elements(self):
        input_data = self.input.get_input_data_batch()
        if input_data is None:
            logging.warning('No InputData. Skipping.')
            return
        self.detection.detect_next_image(input_data['ImageData'])
        detection_data = self.detection.get_detection()
        detections_nms = non_max_suppress_detections(detection_data)
        category_index = create_category_index(FLAGS.label_map_path_detection)
        detection_buttons = filter_one_category_from_detections_nms(detections_nms, category_index, category='btn_floor')
        # TODO: filter out only floor buttons from the detection_data
        return

    def loop(self):
        while True:
            self.recognize_elevator_elements()
