"""A class wrapping the whole project.
TODO: Main class, describe what it does
"""
import os
from absl import logging

from src.elevator_controls_detection.ElementDetection import ElementDetection
from src.elevator_controls_detection.FloorButtonClassification import FloorButtonClassification
from src.input_feed.InputFeed import InputFeed
from src.input_processing.InputProcessing import InputProcessing
from src.main.flags_global import FLAGS
from src.output_processing.OutputProcessing import OutputProcessing
from src.output_visualization.OutputVisualization import OutputVisualization


class RecognizeElevatorElements:
    def __init__(self):
        self.input_feed = InputFeed()
        self.input_process = InputProcessing()
        self.element_detection = ElementDetection()
        self.floor_classification = FloorButtonClassification()
        self.output_process = OutputProcessing(FLAGS.label_map_path_detection,
                                               FLAGS.label_map_path_button_classification)
        self.postprocessing = None  # TODO: these parts are not yet implemented
        self.output_viz = OutputVisualization()
        self.visualization_index = 0

    def recognize_elevator_elements(self):
        """Main loop content of the project."""
        # INPUT
        input_data, input_data_is_valid = self.input_feed.get_input_data_batch()
        if not input_data_is_valid:
            logging.warning('InputData not valid. Skipping.')
            return
        """
        input_data['ImageData'] = self.input_process.equalize_image(input_data['ImageData'])
        if self.input_processing.is_input_image_blurry(input_data['ImageData'].copy()):
            logging.warning('Image considered blurry. Skipping.')
            return
        """
        # DETECTION
        category_index_detection = self.output_process.get_category_index_detection()
        detection_data = self.element_detection.detect_next_image(input_data['ImageData'])
        # TODO: detect irregularities
        if detection_data is None:
            logging.warning('Did not detect anything. Skipping.')
            return
        detections_nms = self.output_process.filter_by_nms(detection_data)
        image_with_detections = self.output_viz.visualize_element_detections(input_data['ImageData'].copy(),
                                                                             detections_nms,
                                                                             category_index_detection)
        # CLASSIFICATION
        detection_buttons = self.output_process.reduce_to_one_category(detections_nms, 'btn_floor')
        detection_buttons_roi = self.output_process.crop_multiple_images_by_bndbox(input_data['ImageData'].copy(),
                                                                                   detection_buttons[
                                                                                       'detection_boxes_nms'])
        if detection_buttons_roi is not None:
            button_classifications = self.floor_classification.classify_next_images(detection_buttons_roi,
                                                                                    input_size=(224, 224))
            category_index_classification = self.output_process.get_category_index_classification()
            btn_highest_classes, btn_highest_scores = self.output_process.filter_highest_classifications(
                button_classifications)
            classifications = {'detection_boxes': detection_buttons['detection_boxes_nms'],
                               'classification_classes': btn_highest_classes,
                               'classification_scores': btn_highest_scores}

            action_area_bndboxes = self.output_process.find_buttons_action_areas(detection_buttons_roi,
                                                                                 detection_buttons[
                                                                                     'detection_boxes_nms'],
                                                                                 input_data['ImageData'],
                                                                                 output_relative_coords=True)

            image_with_action_areas = input_data['ImageData'].copy()
            self.output_viz.draw_bndbox_middle_on_image_from_bndboxes(image_with_action_areas,
                                                                      action_area_bndboxes,
                                                                      use_normalized_coords=True)
            self.output_viz.draw_bndboxes_on_image(image_with_action_areas,
                                                   action_area_bndboxes,
                                                   thickness=2)
            image_with_classification = self.output_viz.visualize_button_classifications(image_with_action_areas,
                                                                                         classifications,
                                                                                         category_index_classification)

        PATH_OUTPUT_TEST_DETECTION = r"C:\Users\cernil\OneDrive - Y Soft Corporation " \
                                     r"a.s\betapresentation_visualisation\outputs\detection"
        PATH_OUTPUT_TEST_CLASSIFICATION = r"C:\Users\cernil\OneDrive - Y Soft Corporation " \
                                          r"a.s\betapresentation_visualisation\outputs\classification"
        self.output_viz.save_as_png(self.input_process.crop_black_outlines(image_with_detections),
                                    PATH_OUTPUT_TEST_DETECTION + os.sep + f"{self.visualization_index}detection.png")
        if detection_buttons_roi is None:
            image_with_classification = input_data['ImageData']
        self.output_viz.save_as_png(self.input_process.crop_black_outlines(image_with_classification),
                                    PATH_OUTPUT_TEST_CLASSIFICATION + os.sep +
                                    f"{self.visualization_index}classification.png")
        self.visualization_index += 1
        return

    def loop(self):
        while True:
            self.recognize_elevator_elements()
