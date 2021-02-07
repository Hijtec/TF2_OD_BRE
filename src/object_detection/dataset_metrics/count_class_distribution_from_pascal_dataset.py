from src.utils.path_utils import get_paths_of_files_with_suffix
import xml.etree.ElementTree as ET


def get_distribution_in_file(xml_file, distribution_dict):
    """
    :param xml_file: PATH to XML file
    :param distribution_dict: dictionary to append results of parsing
    :return: distribution_dict
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        obj_list = root.findall('object')
        for xml_obj in obj_list:
            class_name = xml_obj[0].text
            distribution_dict[class_name] = distribution_dict.get(class_name, 0) + 1
        return distribution_dict
    except Exception as expt:
        print(expt)


def get_class_distribution(annotation_dir_path):
    """
    :param annotation_dir_path: PATH to directory with XML files
    :return: dictionary with N entries per class found
    """
    distribution_dict = dict()
    file_paths_list = get_paths_of_files_with_suffix(annotation_dir_path, '.xml')
    for file_id, xml_file in enumerate(file_paths_list):
        print(f"Processing XML file {xml_file}")
        distribution_dict = get_distribution_in_file(xml_file, distribution_dict)
    return distribution_dict
