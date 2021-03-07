import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
from matplotlib import pyplot as plt
import numpy as np


def __show_image_window(image):
    image.show()


def __show_plt_window(image):
    plt.imshow(image)
    plt.show()


def __concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa + wb
    new_img = np.zeros(shape=(max_height, total_width, 3), dtype=np.uint8)
    new_img[:ha, :wa] = imga
    new_img[:hb, wa:wa + wb] = imgb
    return new_img


def concat_n_images(image_list):
    """
    Combines N color images from a list of images.
    """
    output = None
    i = 0
    for img in image_list:
        if i == 0:
            output = img
        else:
            output = __concat_images(output, img)
        i += 1
    return output


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


def __draw_ellipse_on_image(image_pil, ymin, xmin, ymax, xmax, color='red', thickness=2, use_normalized_coords=True):
    draw = ImageDraw.Draw(image_pil)
    im_width, im_height = image_pil.size
    if use_normalized_coords:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    if thickness > 0:
        draw.ellipse((left, right, top, bottom), fill=color, width=thickness)


def create_image_histogram(img, output_dpi=250):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    plt.figure(dpi=output_dpi)
    plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    canvas = plt.gca().figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    image = data.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image
