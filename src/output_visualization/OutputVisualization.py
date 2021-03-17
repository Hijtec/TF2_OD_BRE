"""A helper class providing methods for processing object detection and classification outputs.
"""
from src.models.research.object_detection.utils.visualization_utils import save_image_array_as_png, \
    draw_bounding_boxes_on_image_array, visualize_boxes_and_labels_on_image_array
from src.utils.visualization_utils import draw_ellipses_in_bndboxes_on_image_array
import numpy as np


class OutputVisualization:
    @staticmethod
    def save_as_png(np_image, output_path):
        """Save a numpy array as PNG image."""
        save_image_array_as_png(np_image, output_path)

    @staticmethod
    def draw_bndboxes_on_image(np_image, bndboxes, color='red', thickness=4, label_lists=tuple()):
        """Draws bounding boxes upon image with provided color, thickness and labels."""
        draw_bounding_boxes_on_image_array(np_image, bndboxes, color, thickness, label_lists)

    @staticmethod
    def draw_ellipses_on_image_from_bndboxes(np_image, bndboxes, color='red', thickness=3, use_normalized_coords=True):
        """Draws ellipses from boxes upon image with provided color and thickness."""
        draw_ellipses_in_bndboxes_on_image_array(np_image, bndboxes, color, thickness, use_normalized_coords)

    @staticmethod
    def draw_bndbox_middle_on_image_from_bndboxes(np_image, bndboxes, color='red', thickness=3, use_normalized_coords=True):
        """Draws middle points from boxes upon image with provided color and thickness."""
        bndboxes_middle_reduced = []
        for bndbox in bndboxes:
            if use_normalized_coords:
                mid_y = (bndbox[0] + bndbox[2])/2
                mid_x = (bndbox[1] + bndbox[3])/2
                dist_y = thickness / np_image.shape[0]
                dist_x = thickness / np_image.shape[1]
                bndbox = [mid_y - dist_y, mid_x - dist_x, mid_y + dist_y, mid_x + dist_x]
            else:
                mid_y = ((bndbox[0] + bndbox[2])/2) / np_image.shape[0]
                mid_x = ((bndbox[1] + bndbox[3])/2) / np_image.shape[1]
                dist_y = thickness / np_image.shape[0]
                dist_x = thickness / np_image.shape[1]
                bndbox = [mid_y - dist_y, mid_x - dist_x, mid_y + dist_y, mid_x + dist_x]
            bndboxes_middle_reduced.append(bndbox)
        bndboxes_middle_reduced = np.array(bndboxes_middle_reduced)
        draw_bounding_boxes_on_image_array(np_image, bndboxes_middle_reduced, color, thickness)

    @staticmethod
    def visualize_on_image(np_image, bndboxes, classes, scores=None, category_index=None,
                           use_normalized_coordinates=False, max_boxes_to_draw=25, min_score_thresh=0.25,
                           agnostic_mode=False, line_thickness=1, groundtruth_box_visualization_color='black',
                           skip_boxes=False, skip_scores=False, skip_labels=False):
        """Draws visualization upon image."""
        visualize_boxes_and_labels_on_image_array(np_image,
                                                  bndboxes,
                                                  classes,
                                                  scores,
                                                  category_index,
                                                  use_normalized_coordinates=use_normalized_coordinates,
                                                  max_boxes_to_draw=max_boxes_to_draw,
                                                  min_score_thresh=min_score_thresh,
                                                  agnostic_mode=agnostic_mode,
                                                  line_thickness=line_thickness,
                                                  skip_boxes=skip_boxes,
                                                  groundtruth_box_visualization_color
                                                  =groundtruth_box_visualization_color,
                                                  skip_scores=skip_scores,
                                                  skip_labels=skip_labels)

    def visualize_element_detections(self, image, detections_nms, category_index_detection):
        image_with_detections = image.copy()
        self.visualize_on_image(image_with_detections,
                                detections_nms['detection_boxes_nms'],
                                detections_nms['detection_classes_nms'].astype(int).tolist(),
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
        self.visualize_on_image(image_with_button_classification,
                                classifications['detection_boxes'],
                                classifications['classification_classes'],
                                classifications['classification_scores'],
                                category_index_classification,
                                use_normalized_coordinates=True,
                                max_boxes_to_draw=25,
                                min_score_thresh=.10,
                                agnostic_mode=False)
        return image_with_button_classification
