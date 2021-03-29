"""A helper class providing methods for processing object detection and classification outputs.
"""
from absl import logging
import numpy as np

from src.utils.detection_utils import create_category_index, get_one_category_from_detections_nms, \
    non_max_suppress_detections, map_predictions_to_output_label_and_sort
from src.utils.image_utils import crop_multiple_by_bndbox
from src.utils.path_utils import check_path_existence
from src.utils.detection_utils import create_category_index_from_list
from src.utils.processing_utils import find_button_action_area_by_hough_circles_nms


class OutputProcessing:
    def __init__(self, label_map_path_detection, label_map_path_classification):
        self.label_map_path_detection = label_map_path_detection
        self.label_map_path_classification = label_map_path_classification

        self.category_index_detection = self._get_category_index(label_map_path_detection)
        self.category_index_classification = self._get_category_index(label_map_path_classification,
                                                                      ignore_background_category=True)

    def _get_category_index(self, label_map_path, ignore_background_category=False):
        check_path_existence(label_map_path, self.__class__.__name__)
        category_index_detection_raw = create_category_index(label_map_path)
        if ignore_background_category:
            index = 0
        else:
            index = 1

        category_index_detection = {}
        for entry in category_index_detection_raw:
            category_index_detection[index] = entry
            index += 1
        return category_index_detection

    def get_category_index_detection(self):
        return self.category_index_detection

    def get_category_index_classification(self):
        return self.category_index_classification

    def reduce_to_one_category(self, detections_nms, category):
        """Reduces detection output to only one specified category."""
        logging.info('Filtered one category from detections.')
        return get_one_category_from_detections_nms(detections_nms, self.category_index_detection, category)

    @staticmethod
    def filter_highest_classifications(button_classifications):
        """Reduces classification to only the highest score label."""
        logging.info('Filtered highest classifications per button.')
        button_classes, button_scores = [], []
        for button in button_classifications:
            class_index = np.argmax(button[0], axis=0)
            button_classes.append(class_index)
            button_scores.append(button[0][class_index])
        return button_classes, button_scores

    @staticmethod
    def crop_multiple_images_by_bndbox(images, bndboxes):
        """Crops multiple images from image by provided bounding boxes."""
        return crop_multiple_by_bndbox(images, bndboxes)

    @staticmethod
    def create_category_index_from_list(label_list):
        """Creates a standardized dictionary of category indexes."""
        return create_category_index_from_list(label_list)

    @staticmethod
    def filter_by_nms(detections, max_output_size=100, iou_threshold=0.5, score_threshold=0.4):
        """Filters the detection output by utilizing Non-Maximum Suppression."""
        logging.info('Non-maximum suppression used on detections.')
        return non_max_suppress_detections(detections, max_output_size, iou_threshold, score_threshold)

    @staticmethod
    def get_classification_label(predictions, labels):
        """Retrieves a list of sorted labels by predictions."""
        logging.info('Retrieved list of most probable labels.')
        return map_predictions_to_output_label_and_sort(predictions, labels)

    @staticmethod
    def find_buttons_action_areas(button_images, button_bndboxes, input_image, output_relative_coords=False):
        """Finds a bounding box of action area of the button."""
        logging.info('Finding buttons action areas.')
        action_area_bndboxes = []
        button_bndboxes_absolute = []

        for entry in button_bndboxes:
            bndbox_absolute = [entry[0] * input_image.shape[0],
                               entry[1] * input_image.shape[1],
                               entry[2] * input_image.shape[0],
                               entry[3] * input_image.shape[1]]
            button_bndboxes_absolute.append(bndbox_absolute)

        for button_image, button_bndbox in zip(button_images, button_bndboxes_absolute):
            action_area_bndboxes.append(find_button_action_area_by_hough_circles_nms(button_image, button_bndbox))

        if output_relative_coords is True:
            button_action_areas_relative = []
            for button in action_area_bndboxes:
                for entry in button:
                    bndbox_relative = [entry[0] / input_image.shape[0],
                                       entry[1] / input_image.shape[1],
                                       entry[2] / input_image.shape[0],
                                       entry[3] / input_image.shape[1]]
                    button_action_areas_relative.append(bndbox_relative)
            button_action_areas_relative = np.array(button_action_areas_relative)
            return button_action_areas_relative

        action_area_bndboxes = np.array(action_area_bndboxes)
        return action_area_bndboxes

    @staticmethod
    def relabel_wrong_buttons(button_bndboxes, button_classifications):
        """Relabels wrong buttons based on their position numbering and ambiguous classifications."""
        pass
        # TODO: implement
