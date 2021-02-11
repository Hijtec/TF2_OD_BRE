import tensorflow as tf
import numpy as np

from src.models.research.object_detection.utils.label_map_util import load_labelmap, convert_label_map_to_categories


def filter_one_category_from_detections_nms(detections_nms, category_index, category):
    ids_to_find = np.array([])
    indexes_to_grab = np.array([])
    detections_nms_filtered = {}
    for index_dict in category_index:
        if index_dict['name'] == category:
            if index_dict['id'] not in ids_to_find:
                ids_to_find = np.append(ids_to_find, index_dict['id'])
    i = 0
    for detected_class_id in detections_nms['detection_classes_nms']:
        if int(detected_class_id) in ids_to_find:
            indexes_to_grab = np.append(indexes_to_grab, i)
        i += 1
    for key, np_array_data in detections_nms.items():
        detections_nms_filtered[key] = np_array_data[indexes_to_grab.astype('uint8')]
    return detections_nms_filtered


def create_category_index(label_map_path):
    label_map = load_labelmap(label_map_path)
    max_num_classes = max(item.id for item in label_map.item)
    category_index = convert_label_map_to_categories(label_map, max_num_classes, True)
    return category_index


def non_max_suppress_detections(detections, max_output_size=100, iou_threshold=0.5, score_threshold=0.3):
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy()
    detection_scores = detections['detection_scores'][0].numpy()
    indexes = np.array(tf.image.non_max_suppression(
            detection_boxes,
            detection_scores,
            max_output_size,
            iou_threshold,
            score_threshold))

    detection_boxes_nms = detection_boxes[indexes]
    detection_classes_nms = detection_classes[indexes]
    detection_scores_nms = detection_scores[indexes]

    return {'detection_boxes_nms': detection_boxes_nms,
            'detection_classes_nms': detection_classes_nms,
            'detection_scores_nms': detection_scores_nms}
