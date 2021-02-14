"""A class wrapping the whole project.
TODO: Main class, describe what it does
"""
from src.elevator_controls_detection.ElementDetection import ElementDetection
from src.elevator_controls_detection.FloorButtonClassification import FloorButtonClassification
from src.input_feed.InputFeed import InputFeed
from src.output_processing.OutputProcessing import OutputProcessing
from src.output_visualization.OutputVisualization import OutputVisualization
from src.main.flags_global import FLAGS
from absl import logging

from src.utils.detection_utils import create_category_index_from_list
from src.utils.image_utils import crop_multiple_images_by_bndbox

import numpy as np


class RecognizeElevatorElements:
    def __init__(self):
        self.input = InputFeed()
        self.detection = ElementDetection()
        self.classification = FloorButtonClassification()
        self.processing = OutputProcessing(FLAGS.label_map_path_detection)
        self.postprocessing = None  # TODO: these parts are not yet implemented
        self.visualization = OutputVisualization()

    def recognize_elevator_elements(self):
        """Main loop content of the project."""
        input_data = self.input.get_input_data_batch()
        if input_data is None:
            logging.warning('No InputData. Skipping.')
            return
        detection_data = self.detection.detect_next_image(input_data['ImageData'])
        detections_nms = self.processing.filter_by_nms(detection_data)
        category_index = self.processing.category_index_detection
        index = 1
        category_index_detection = {}
        for entry in category_index:
            category_index_detection[str(index)] = entry
            index += 1
        image_with_detections = self.visualize_element_detections(input_data['ImageData'].copy(),
                                                                  detections_nms,
                                                                  category_index_detection)

        detection_buttons = self.processing.filter_one_category(detections_nms, 'btn_floor')
        detection_buttons_roi = crop_multiple_images_by_bndbox(input_data['ImageData'],
                                                               detection_buttons['detection_boxes_nms'])
        button_classifications = self.classification.classify_next_images(detection_buttons_roi, input_size=(224, 224))
        button_labels = [-2, -1, 0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 3, 4, 5, 6, 7, 8, 9]

        category_index_classification = create_category_index_from_list(button_labels)
        button_classes, button_scores = [], []
        for button in button_classifications:
            class_index = np.argmax(button[0], axis=0)
            button_classes.append(class_index)
            button_scores.append(button[0][class_index])
        classifications = {'detection_boxes': detection_buttons['detection_boxes_nms'],
                           'classification_classes': button_classes,
                           'classification_scores': button_scores}
        image_with_classification = self.visualize_button_classifications(input_data['ImageData'],
                                                                          classifications,
                                                                          category_index_classification)
        return

    def loop(self):
        while True:
            self.recognize_elevator_elements()

    def visualize_element_detections(self, image, detections_nms, category_index_detection):
        image_with_detections = image.copy()
        self.visualization.visualize_on_image(image_with_detections,
                                              detections_nms['detection_boxes_nms'],
                                              detections_nms['detection_classes_nms'].astype(int),
                                              detections_nms['detection_scores_nms'],
                                              category_index_detection,
                                              use_normalized_coordinates=True,
                                              line_thickness=1,
                                              max_boxes_to_draw=50,
                                              min_score_thresh=.20,
                                              agnostic_mode=False)
        return image_with_detections

    def visualize_button_classifications(self, image, classifications, category_index_classification):
        image_with_button_classification = image.copy()
        self.visualization.visualize_on_image(image_with_button_classification,
                                              classifications['detection_boxes'],
                                              classifications['classification_classes'],
                                              classifications['classification_scores'],
                                              category_index_classification,
                                              use_normalized_coordinates=True,
                                              max_boxes_to_draw=25,
                                              min_score_thresh=.00,
                                              agnostic_mode=False)
        return image_with_button_classification
