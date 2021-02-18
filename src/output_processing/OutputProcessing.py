"""A helper class providing methods for processing object detection and classification outputs.
"""
from absl import logging
import numpy as np

from src.utils.detection_utils import create_category_index, get_one_category_from_detections_nms, \
    non_max_suppress_detections, map_predictions_to_output_label_and_sort
from src.utils.image_utils import crop_multiple_images_by_bndbox
from src.utils.path_utils import check_path_existence
from src.utils.detection_utils import create_category_index_from_list


class OutputProcessing:
    def __init__(self, label_map_path):
        self.label_map_path = label_map_path
        self.category_index_detection = self.__get_category_index_detection(label_map_path)

    def __get_category_index_detection(self, label_map_path):
        check_path_existence(label_map_path, self.__class__.__name__)
        category_index_detection_raw = create_category_index(label_map_path)
        index = 1
        category_index_detection = {}
        for entry in category_index_detection_raw:
            category_index_detection[index] = entry
            index += 1
        return category_index_detection

    def get_category_index_detection(self):
        return self.category_index_detection

    def reduce_to_one_category(self, detections_nms, category):
        """Reduces detection output to only one specified category."""
        logging.info('Filtered one category from detections.')
        return get_one_category_from_detections_nms(detections_nms, self.category_index_detection, category)

    @staticmethod
    def filter_highest_classifications(button_classifications):
        button_classes, button_scores = [], []
        for button in button_classifications:
            class_index = np.argmax(button[0], axis=0)
            button_classes.append(class_index)
            button_scores.append(button[0][class_index])
        return button_classes, button_scores

    @staticmethod
    def crop_multiple_images_by_bndbox(images, bndboxes):
        """Crops multiple images from image by provided bounding boxes."""
        return crop_multiple_images_by_bndbox(images, bndboxes)

    @staticmethod
    def create_category_index_from_list(label_list):
        """Creates a standardized dictionary of category indexes."""
        return create_category_index_from_list(label_list)

    @staticmethod
    def filter_by_nms(detections, max_output_size=100, iou_threshold=0.5, score_threshold=0.3):
        """Filters the detection output by utilizing Non-Maximum Suppression."""
        logging.info('Non-maximum suppression used on detections.')
        return non_max_suppress_detections(detections, max_output_size, iou_threshold, score_threshold)

    @staticmethod
    def get_classification_label(predictions, labels):
        """Retrieves a list of sorted labels by predictions."""
        logging.info('Retrieved list of most probable labels.')
        return map_predictions_to_output_label_and_sort(predictions, labels)
