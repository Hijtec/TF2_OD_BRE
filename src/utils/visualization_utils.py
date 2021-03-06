import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
from matplotlib import pyplot as plt
import numpy as np


def __show_image_window(image):
    image.show()


def __show_plt_window(image):
    plt.imshow(image)
    plt.show()


def show_multiple_images_side_by_side(*images):
    resulting_image = []
    for img in images:
        np.vstack(resulting_image, img)
    __show_plt_window(resulting_image)

def draw_ellipses_in_bndboxes_on_image_array(image, boxes, color='red', thickness=3, use_normalized_coords=True):
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('Input must be of size [N, 4]')
    image_pil = Image.fromarray(image)
    for box in boxes:
        __draw_ellipse_on_image(image_pil, box[0], box[1], box[2], box[3], color, thickness, use_normalized_coords)
    np.copyto(image, np.array(image_pil))

def __draw_ellipse_on_image(image_pil, ymin, xmin, ymax, xmax, color='red', thickness = 2, use_normalized_coords=True):
    draw = ImageDraw.Draw(image_pil)
    im_width, im_height = image_pil.size
    if use_normalized_coords:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    if thickness > 0:
        draw.ellipse((left, right, top, bottom), fill=color, width=thickness)