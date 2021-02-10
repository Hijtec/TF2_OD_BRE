"""A preprocessing module for button recognition.
This module contains functions needed for preprocessing an image:
    grayscaling
    median filtering
    canny edge detection
    candidate ROI extraction
    size filtering
"""
import cv2
import imutils
import numpy as np

from imutils import contours


def adjust_brightness_dynamic(image, brightness=0.0, contrast=0.0):
    """Adjust the brightness using imutils utility function."""
    return imutils.adjust_brightness_contrast(image, brightness=brightness, contrast=contrast)


def grayscale(image):
    """Grayscales the image."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def median_filter(image, kernel_size=3):
    """Applies median filter on image."""
    return cv2.medianBlur(image, kernel_size)


def canny_edge_extraction(image, sigma=0.33, dilate_iterations=2, erode_iterations=1):
    """Applies auto_canny imutils utility to get image edges."""
    edges_raw = imutils.auto_canny(image, sigma)
    edges_dilated = cv2.dilate(edges_raw, None, iterations=dilate_iterations)
    edges_eroded = cv2.erode(edges_dilated, None, iterations=erode_iterations)
    return edges_eroded


def contours_extraction_sort(image, edges):
    """Finds contours and sorts them out."""
    orig_label = None
    sorted_label = None

    clone = image.copy()
    cnts = cv2.findContours(
        edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = imutils.grab_contours(cnts)
    for (i, cnt) in enumerate(cnts):
        orig_label = contours.label_contour(image, cnt, i, color=[240, 0, 159])
    (cnts_ordered, __) = contours.sort_contours(cnts, method="top-to-bottom")
    for (i, cnt) in enumerate(cnts_ordered):
        sorted_label = contours.label_contour(clone, cnt, i, color=[240, 0, 50])
    return orig_label, sorted_label, cnts, cnts_ordered


def box_contours(image, cnts):
    """Gives contours bounding boxes and draws them upon image."""
    boxes = []
    for __, cnt in enumerate(cnts):
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        boxes.append(box)
        image = cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
    return image, boxes


def bbox_filter_by_dimension_limits(image, boxes, size_lim=None, width_lim=None, height_lim=None, ratio_wh_lim=None):
    """Based on criteria filters the given bounding boxes and draws them upon image. Limits given are lists of len 2"""
    boxes_filtered = []
    for box in boxes:
        x_1, y_1 = box[0]
        x_2, y_2 = box[1]
        x_3, y_3 = box[2]
        x_4, y_4 = box[3]
        dx_12 = abs(x_1 - x_2)
        dy_12 = abs(y_1 - y_2)
        dx_23 = abs(x_2 - x_3)
        dy_23 = abs(y_2 - y_3)
        dy_14 = abs(y_1 - y_4)

        l12 = dx_12 + dy_12
        l23 = dx_23 + dy_23

        if dy_12 <= dy_14:
            width = l12
            height = l23
        else:
            width = l23
            height = l12

        size = l12 * l23
        wh_ratio = width / height

        def between(limit, value):
            """Returns True if the value is between the limits."""
            if limit is not None:
                return limit[0] <= value <= limit[1]
            else:
                return True

        # CRITERIA
        size_criteria = between(size_lim, size)
        height_criteria = between(height_lim, height)
        width_criteria = between(width_lim, width)
        ratio_criteria = between(ratio_wh_lim, wh_ratio)
        # VALID BOX?
        if size_criteria and height_criteria and width_criteria and ratio_criteria:
            boxes_filtered.append(box)

    for box in boxes_filtered:
        image = cv2.drawContours(image, [box], 0, (255, 0, 0), 2)
    return image, boxes_filtered
