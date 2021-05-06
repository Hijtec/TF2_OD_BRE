"""A script providing global variables for scripts.
This module contains all flags used in the project.
"""

from absl import flags

# Flag names are globally defined!  So in general, we need to be
# careful to pick names that are unlikely to be used by other libraries.
# If there is a conflict, we'll get an error at import time.
flags.DEFINE_string('f', '', 'kernel')  # workaround due to JupyterNotebook
#####
# INPUT FEED FLAGS BEGIN
# common
flags.DEFINE_enum('image_input_mode', 'folder', ['camera', 'video_per_frame', 'video_async', 'folder'], 'Source of '
                                                                                                        'image data.')
flags.DEFINE_enum('location_input_mode', 'pos_constant', ['pos_constant', 'reserved'], 'Source of location data.')

# image_feed_camera
flags.DEFINE_integer('camera_device_used', 0, 'An index of the used camera device', lower_bound=0)
# image_feed_camera
flags.DEFINE_string('image_input_folder_path', r"C:\Users\cernil\OneDrive - Y Soft Corporation "
                                               r"a.s\DeepLearningBlackbox\test\test_images_elevator_control_panel",
                    'PATH to the folder with images, images passed recursively.')
# image_feed_video
flags.DEFINE_string('image_input_video_path', r"C:\Users\cernil\OneDrive - Y Soft Corporation "
                                              r"a.s\DeepLearningBlackbox\havelka\display_extraction_video"
                                              r"\IBC_STEF_BIEB_DISPLAY_VIDEO.mp4", 'PATH to the folder with video, '
                                                                                   'used while '
                                                                                   'image_input_mode=video.')
# position_feed_constant
flags.DEFINE_enum('robot_position', 'elevator', ['elevator', 'hall', 'reserved'], 'Current location of the robot.')

# INPUT FEED FLAGS END
#####
# ELEVATOR CONTROLS DETECTION FLAGS BEGIN
# elevator element detection
flags.DEFINE_string('detector_elements_model_path', r"C:\Users\cernil\OneDrive - Y Soft Corporation "
                                                    r"a.s\DeepLearningBlackbox\test\output\saved_model",
                    'PATH to a SavedModel folder capable of detection of elevator elements.')
flags.DEFINE_enum('detection_elements_model_type', 'tf2',
                  ['tf2', 'reserved'], 'Type of detection model used - RESERVED.')
flags.DEFINE_string('label_map_path_detection',
                    r"C:\Users\cernil\OneDrive - Y Soft Corporation "
                    r"a.s\DeepLearningBlackbox\test\output\pascal_label_map.pbtxt",
                    'PATH to the label_map.txt | label_map.pbtxt file for detection.')
# floor button classification
flags.DEFINE_string('classification_floor_button_model_path', r"C:\Users\cernil\OneDrive - Y Soft Corporation "
                                                              r"a.s\DeepLearningBlackbox\button_classifier",
                    'PATH to a SavedModel folder capable of classification of elevator floor buttons.')
flags.DEFINE_enum('classification_floor_button_model_type', 'keras',
                  ['keras', 'reserved'], 'Type of classification model used - RESERVED.')
flags.DEFINE_string('label_map_path_button_classification',
                    r"C:\Users\cernil\OneDrive - Y Soft Corporation "
                    r"a.s\DeepLearningBlackbox\test\output\pascal_label_map_classification.pbtxt",
                    'PATH to the label_map.txt | label_map.pbtxt file for button classification .')
# ELEVATOR CONTROLS DETECTION FLAGS END
#####
# OUTPUT VISUALIZATION FLAGS BEGIN
flags.DEFINE_string('output_image_save_detection',
                    r"C:\Users\cernil\OneDrive - Y Soft Corporation "
                    r"a.s\betapresentation_visualisation\outputs\detection",
                    'PATH to the output folder to save detections.')

flags.DEFINE_string('output_image_save_classification',
                    r"C:\Users\cernil\OneDrive - Y Soft Corporation "
                    r"a.s\betapresentation_visualisation\outputs\classification",
                    'PATH to the output folder to save detections.')
# OUTPUT VISUALIZATION FLAGS END
#####
FLAGS = flags.FLAGS
