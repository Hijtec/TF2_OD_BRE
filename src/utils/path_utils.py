from glob import glob
from os import path
from os import sep as sep


def get_paths_of_files_with_suffix(dir_path, file_suffix):
    """
    :param dir_path: PATH to directory with files
    :param file_suffix: ex.: ".jpeg"
    :return: glob object containing generalized path
    """
    search_pattern = path.join(dir_path, f'*{file_suffix}')
    return glob(search_pattern)


def extract_image_name(annotation_path, img_suffix=None):
    """
    :param annotation_path: PATH to annotation xml file
    :param img_suffix: string of image type
    :return: string of image name with img_suffix
    """
    ann_name = path.basename(annotation_path).split('.')[0]
    img_name = f'{ann_name}{img_suffix}'
    return img_name


def convert_annotation_path_to_image_path(annotation_path, image_dir_path, img_suffix='.png'):
    """
    :param annotation_path: PATH to annotation file
    :param image_dir_path: PATH to directory with images from annotation file
    :param img_suffix: string of image filetype ex.: ".png"
    :return: PATH to image
    """
    image_name = extract_image_name(annotation_path, img_suffix)
    return path.join(image_dir_path, image_name)


def path_exists(input_path):
    if path.exists(input_path):
        return True
    else:
        return False


def check_path_existence(input_path, object_name):
    if path_exists(input_path) is False:
        raise ValueError(f"Path used in method: {object_name} does not exist.")
    return input_path


def force_separator_as_path_end(input_path):
    return input_path if input_path[-1] == sep else input_path + sep
