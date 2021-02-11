import numpy as np


def crop_image_by_bndbox(image, bndbox, normalized_coordinates=True):
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
    temp_list = None
    for bndbox in bndboxes:
        cropped_image = crop_image_by_bndbox(image, bndbox)
        if temp_list is None:
            temp_list = [cropped_image]
            continue
        temp_list.append(cropped_image)
    return temp_list
