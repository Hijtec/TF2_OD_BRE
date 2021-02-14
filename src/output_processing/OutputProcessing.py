"""A helper class providing methods for processing object detection and classification outputs.
"""
from src.utils.detection_utils import create_category_index, filter_one_category_from_detections_nms, \
    non_max_suppress_detections, map_predictions_to_output_label_and_sort
from absl import logging


class OutputProcessing:
    def __init__(self, label_map_path):
        self.label_map_path = label_map_path
        self.category_index_detection = create_category_index(label_map_path)

    def filter_one_category(self, detections_nms, category):
        logging.info('Filtered one category from detections.')
        return filter_one_category_from_detections_nms(detections_nms, self.category_index_detection, category)

    def filter_by_nms(self, detections, max_output_size=100, iou_threshold=0.5, score_threshold=0.3):
        logging.info('Non-maximum suppression used on detections.')
        return non_max_suppress_detections(detections, max_output_size, iou_threshold, score_threshold)

    def get_classification_label(self, predictions, labels):
        logging.info('Retrieved list of most probable labels.')
        return map_predictions_to_output_label_and_sort(predictions, labels)
