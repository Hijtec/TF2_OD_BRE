import os
from src.utils.path_utils import check_path_existence
import cv2


def generate_x_images_from_folder(folder_path, gen_x=50, start=0, min_move=0.4):
    import random
    check_path_existence(folder_path, 'generate_x_images_from_folder')
    n_examples = len(os.listdir(folder_path))
    sample_slice_size = int(n_examples/gen_x)
    min_move_in_folder = int(min_move * sample_slice_size)
    samples = []
    index = start
    for _ in range(gen_x):
        index += random.randint(min_move_in_folder, sample_slice_size)
        image_read = cv2.imread(os.path.join(folder_path, os.listdir(folder_path)[index]))
        if image_read is not None:
            samples.append(image_read)
    return samples
