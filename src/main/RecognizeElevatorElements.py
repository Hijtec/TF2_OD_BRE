"""A class wrapping the whole project.
TODO: Main class, describe what it does
"""
from src.elevator_controls_detection.ElementDetection import ElementDetection
from src.elevator_controls_detection.FloorButtonClassification import FloorButtonClassification
from src.input_feed.InputFeed import InputFeed
from src.main.flags_global import FLAGS
from absl import logging

from src.utils.detection_utils import non_max_suppress_detections, filter_one_category_from_detections_nms, create_category_index
from src.utils.image_utils import crop_multiple_images_by_bndbox
from matplotlib.pyplot import imshow, show
import tensorflow as tf
import numpy as np


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
        detections_nms = non_max_suppress_detections(detection_data)
        category_index = create_category_index(FLAGS.label_map_path_detection)
        detection_buttons = filter_one_category_from_detections_nms(detections_nms, category_index, category='btn_floor')
        detection_buttons_roi = crop_multiple_images_by_bndbox(input_data['ImageData'],
                                                               detection_buttons['detection_boxes_nms'])
        button_classifications = []
        for button_roi in detection_buttons_roi:
            button_roi_tensor = tf.convert_to_tensor(np.expand_dims(button_roi, 0), dtype=tf.float32)
            button_roi_tensor_resized = tf.image.resize_with_pad(button_roi_tensor, 224, 224, method=tf.image.ResizeMethod.BICUBIC)
            imshow(np.squeeze(button_roi_tensor_resized.numpy().astype(tf.int32)))
            show()
            self.classification.classify_next_tensor(button_roi_tensor_resized)
            button_classification = self.classification.get_classification()
            button_classifications.append(button_classification)
        return

    def loop(self):
        while True:
            self.recognize_elevator_elements()
