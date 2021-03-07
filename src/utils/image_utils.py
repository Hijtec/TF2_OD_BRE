import numpy as np
from PIL import Image
import cv2


def convert_pil_to_np_array(image):
    """
    :param image: Pillow Image object
    :return: ndarray representation of input image
    """
    return np.asarray(image)


def convert_np_array_to_pil(np_array):
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
    img_np_cropped = crop_by_bndbox(img_np_to_crop, [x0, y0, x1, y1], normalized_coordinates=False)
    return img_np_cropped


def crop_by_bndbox(image, bndbox, normalized_coordinates=True):
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


def crop_multiple_by_bndbox(image, bndboxes):
    """
    :param image: ndarray representing an image to be cropped from
    :param bndboxes: list of lists/tuples of [y0, x0, y1, x1] bounding box coordinates
    :return: list of ndarrays, crops from image of shape (width, height, channels)
    """
    crops = None
    for bndbox in bndboxes:
        cropped_image = crop_by_bndbox(image, bndbox)
        if crops is None:
            crops = [cropped_image]
            continue
        crops.append(cropped_image)
    return crops


def grayscale(image):
    """Grayscales an image."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def grayscale_to_colour(image):
    """Converts grayscale image to equivalent color one."""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def gaussian_blur(image, kernel_size=(5, 5), std_distribution=1.0):
    """Blurs an image with the gaussian filter."""
    return cv2.GaussianBlur(image, kernel_size, std_distribution)


def median_blur(image, kernel_size=5):
    """Blurs an image with the median filter."""
    return cv2.medianBlur(image, kernel_size)


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, intensity=1.0, threshold=0):
    """Produce a sharpened version of the image, using an unsharp mask.
    :param image: ndarray of shape (width, height, channels)
    :param kernel_size: tuple with the size of kernel of shape (x, y)
    :param sigma: float standard distribution used in gaussian blur
    :param intensity: float representing the intensity of sharpening
    :param threshold: float threshold value for low contrast mask. If 0, low contrast mask not used
    :return: ndarray of same shape as param image but sharpened
    """
    blurred = gaussian_blur(image.copy(), kernel_size, std_distribution=sigma)
    sharpened = float(intensity + 1) * image - float(intensity) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def auto_canny(image, sigma=0.33):
    """Produce an edges image version of the image"""
    v = np.median(image)  # compute the median of the single channel pixel intensities
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)  # apply automatic Canny edge detection using the computed median
    return edged


def enhance_edges(img, kernel_intensity=17):
    blur = cv2.bilateralFilter(img, 5, 75, 75)
    ki = kernel_intensity
    kernel_sharp = np.array((
         [-2, -2, -2],
         [-2, ki, -2],
         [-2, -2, -2]), dtype='int')
    im = cv2.filter2D(blur, -1, kernel_sharp)
    return im


def equalize(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


