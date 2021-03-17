from src.utils.image_utils import grayscale, unsharp_mask, median_blur
from absl import logging
import cv2
import numpy as np


def find_button_action_area_by_hough_circles_nms(button_image, button_bndbox_in_input_image=(0, 0)):
    """
    :param button_image: ndarray of shape (width, height, channels)
    :param button_bndbox_in_input_image: tuple of button bounding box origin (x, y), used for absolute coords ROI.
    :return: ndarray ROI of the action area in button_image
    """
    gray = grayscale(button_image)
    blurred = median_blur(gray, 3)
    sharpened_edges = unsharp_mask(blurred, kernel_size=(5, 5), sigma=5.0, intensity=5.0, threshold=100)
    sharpened_edges_blurred = median_blur(sharpened_edges, 3)
    circles = __find_hough_circles(sharpened_edges_blurred)
    if circles is None:
        logging.info('Did not found any circles.')
        return None
    circles = np.round(circles[0, :]).astype('int')
    bboxes = []
    for (x, y, r) in circles:
        # surround circles with bounding boxes
        bboxes.append([y - r, x - r, y + r, x + r])
    bboxes_np = np.array(bboxes)
    bboxes_nms = non_max_suppression_slow(bboxes_np, 0.0)
    # move the bounding box to get absolute coordinates relative to image from which the button roi was extracted
    bboxes_absolute_coords = []
    for box in bboxes_nms:
        bboxes_absolute_coords.append(move_bndbox_by_xy(box, button_bndbox_in_input_image))

    return bboxes_absolute_coords


def __find_hough_circles(image, circles_to_find=3, accumulator_start=1.0, minimum_distance=1, accumulator_threshold=60):
    """
    :param image: ndarray of shape (width, height, channels)
    :param circles_to_find: int number of circles to find in image
    :param accumulator_start: float starting value of accumulator in hough circles detection
    :param minimum_distance: int minimum distance between centres of detected circles, in pixels?
    :param accumulator_threshold: Accumulator threshold for the circle centers at the detection stage.
    The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values,
     will be returned first.
    :return: numpy array containing arrays of list objects of content [x_coord, y_coord, radius]
    """
    accumulator = accumulator_start
    n_circles_found = 0
    circles = None
    while (n_circles_found < circles_to_find) and (accumulator <= 5):
        circles = cv2.HoughCircles(image=image,
                                   method=cv2.HOUGH_GRADIENT,
                                   dp=accumulator,
                                   minDist=minimum_distance,
                                   param1=None,
                                   param2=accumulator_threshold,
                                   minRadius=int(image.shape[0] * 0.15),
                                   maxRadius=int(image.shape[0] * 0.6))
        accumulator += 0.02
        if circles is not None:
            n_circles_found = circles.shape[1]
    return circles


def remove_non_overlapping_bndboxes(boxes):
    """Removes bounding boxes that are not overlapping with anything"""
    # TODO: implement this method


def non_max_suppression_slow(boxes, overlap_threshold=0.0):
    """Non-Maximum Suppression on given bounding boxes and given overlap threshold."""
    if len(boxes) == 0:
        return []  # if there are no boxes, return an empty list
    pick = []  # initialize the list of picked indexes
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]  # grab the coordinates of the bounding boxes
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    indexes = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes list
    while len(indexes) > 0:
        # grab the last index in the indexes list, add the index value to the list of picked indexes,
        # then initialize the suppression list (i.e. indexes that will be deleted) using the last index
        last = len(indexes) - 1
        i = indexes[last]
        pick.append(i)
        suppress = [last]

        # loop over all indexes in the indexes list
        for pos in range(0, last):
            j = indexes[pos]  # grab the current index

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # if there is sufficient overlap, suppress the current bounding box
            if overlap > overlap_threshold:
                suppress.append(pos)

        # delete all indexes from the index list that are in the
        # suppression list
        indexes = np.delete(indexes, suppress)
    return boxes[pick]  # return only the bounding boxes that were picked


def move_bndbox_by_xy(bndbox, yx):
    """Move bounding box by x and y coordinates."""
    y, x = yx[0], yx[1]
    bndbox[0] += y
    bndbox[1] += x
    bndbox[2] += y
    bndbox[3] += x
    return bndbox
