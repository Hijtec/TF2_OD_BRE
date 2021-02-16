import numpy as np
from PIL import Image


def convert_image_to_np_array(image):
    """
    :param image: Pillow Image object
    :return: ndarray representation of input image
    """
    return np.asarray(image)


def convert_np_array_to_image(np_array):
    """
    :param np_array: ndarray of shape (height, width, channels)
    :return: Pillow Image object
    """
    return Image.fromarray(np_array)


def crop_to_area_above_threshold(img_np_to_crop, threshold=0):
    """
    :param img_np_to_crop: ndarray including area to be cropped above threshold
    :param threshold: int thresholding value to filter
    :return: ndarray representing a cropped image
    """
    img_grayscale = Image.fromarray(img_np_to_crop).convert('L')
    img_np_grayscale = np.asarray(img_grayscale)
    mask = img_np_grayscale > threshold  # Mask of pixels above threshold
    coords = np.argwhere(mask)  # Coordinates of non-threshold pixels

    # Bounding box of non-threshold pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1  # slices are exclusive at the top
    img_np_cropped = crop_image_by_bndbox(img_np_to_crop, [x0, y0, x1, y1], normalized_coordinates=False)
    return img_np_cropped


def crop_image_by_bndbox(image, bndbox, normalized_coordinates=True):
    """
    :param image: ndarray representing an image to be cropped of shape (width, height, channels)
    :param bndbox: list/tuple of [y0, x0, y1, x1] bounding box coordinates
    :param normalized_coordinates: bool whether the bndbox coordinates are normalized
    :return: ndarray, a crop from image of shape (width, height, channels)
    """
    y0n, x0n, y1n, x1n = bndbox[0], bndbox[1], bndbox[2], bndbox[3]
    width, height = image.shape[1], image.shape[0]
    if normalized_coordinates is True:
        x0, x1 = int(x0n * width), int(x1n * width)
        y0, y1 = int(y0n * height), int(y1n * height)
    else:
        x0, x1 = x0n, x1n
        y0, y1 = y0n, y1n

    cropped_img = image[y0:y1, x0:x1, :]
    return cropped_img


def crop_multiple_images_by_bndbox(image, bndboxes):
    """
    :param image: ndarray representing an image to be cropped from
    :param bndboxes: list of lists/tuples of [y0, x0, y1, x1] bounding box coordinates
    :return: list of ndarrays, crops from image of shape (width, height, channels)
    """
    crops = None
    for bndbox in bndboxes:
        cropped_image = crop_image_by_bndbox(image, bndbox)
        if crops is None:
            crops = [cropped_image]
            continue
        crops.append(cropped_image)
    return crops
