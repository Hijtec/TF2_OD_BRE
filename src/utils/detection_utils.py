import tensorflow as tf
import numpy as np

from src.models.research.object_detection.utils.label_map_util import load_labelmap, convert_label_map_to_categories


def get_one_category_from_detections_nms(detections_nms, category_index, category):
    """Gets one category from detections.
    :param detections_nms: dictionary of outputs from detection, preferably already filtered by NMS
    :param category_index: dictionary of standardized category_index type mapping ids to class_name
    :param category: string - class_name to get all occurrences of
    :return:
    """
    ids_to_find = np.array([])
    indexes_to_get = np.array([])
    detections_nms_one_category = {}
    for key, value in category_index.items():
        if value['name'] == category:
            if value['id'] not in ids_to_find:
                ids_to_find = np.append(ids_to_find, value['id'])
    i = 0
    for detected_class_id in detections_nms['detection_classes_nms']:
        if int(detected_class_id) in ids_to_find:
            indexes_to_get = np.append(indexes_to_get, i)
        i += 1
    for key, np_array_data in detections_nms.items():
        detections_nms_one_category[key] = np_array_data[indexes_to_get.astype('uint8')]
    return detections_nms_one_category


def create_category_index(label_map_path):
    """
    :param label_map_path: PATH to "label_map".txt|.pbtxt file
    :return: dictionary of standardized category_index type
    """
    label_map = load_labelmap(label_map_path)
    max_num_classes = max(item.id for item in label_map.item)
    category_index = convert_label_map_to_categories(label_map, max_num_classes, True)
    return category_index


def create_category_index_from_list(classname_list, label_offset=0):
    """
    :param classname_list: list of string entries - labels
    :param label_offset: int offsetting the labels by x
    :return: dictionary of standardized category_index type
    """
    category_index = {}
    index = 0 + label_offset
    for classname in classname_list:
        category_index[index] = {'id': index, 'name': str(classname)}
        index += 1
    return category_index


def non_max_suppress_detections(detections, max_output_size=100, iou_threshold=0.5, score_threshold=0.3):
    """Filters detections by Non-Max Suppression"""
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


def map_predictions_to_output_label_and_sort(predictions, labels):
    """Maps predictions to labels.
    Eg.: predictions = [1e-3,   3e-2,   2e-3,   0.96,    2e-3]
    Eg.: labels      = ['bike', 'dog',  'cat',  'fly',  'house']
    returns ['fly', 'dog', 'house', 'cat', 'bike']
    :param predictions: Ndarray of shape (1, x) with prediction scores
    :param labels: Ndarray of shape (1, x)
    :return: List of sorted labels by predictions
    """
    predictions_sorted, labels_sorted = zip(*sorted(zip(predictions, labels), reverse=True))
    return predictions_sorted, labels_sorted
