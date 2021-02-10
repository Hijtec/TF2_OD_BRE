"""A script providing global variables for scripts.
This module contains all flags used in the project.
"""

from absl import flags
# from absl import app
# TODO: run main as app for flags to be loaded and parsed

# Flag names are globally defined!  So in general, we need to be
# careful to pick names that are unlikely to be used by other libraries.
# If there is a conflict, we'll get an error at import time.
"""
flags.DEFINE_string('name', 'Jane Random', 'Your name.')
flags.DEFINE_integer('age', None, 'Your age in years.', lower_bound=0)
flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
flags.DEFINE_enum('job', 'running', ['running', 'stopped'], 'Job status.')
"""
#####
# INPUT FEED FLAGS START
# common
flags.DEFINE_enum('image_input_mode', 'camera', ['camera', 'video', 'folder'], 'Source of image data.')
flags.DEFINE_enum('location_input_mode', 'pos_constant', ['pos_constant', 'reserved'], 'Source of location data.')

# image_feed_camera
flags.DEFINE_integer('camera_device_used', 0, 'An index of the used camera device', lower_bound=0)
# image_feed_camera
flags.DEFINE_string('image_input_folder_path', None, 'PATH to the folder with images, images passed recursively.')
# image_feed_video
flags.DEFINE_string('image_input_folder_path', None, 'PATH to the folder with images, images passed recursively.')
# position_feed_constant
flags.DEFINE_enum('robot_position', 'elevator', ['elevator', 'hall', 'reserved'], 'Current location of the robot.')

# INPUT FEED FLAGS END
#####
# ELEVATOR CONTROLS DETECTION FLAGS START
# elevator element detection
flags.DEFINE_string('detector_elements_model_path', None,
                    'PATH to a SavedModel file capable of detection of elevator elements.')
flags.DEFINE_enum('detection_elements_model_type', 'tf2',
                  ['tf2', 'reserved'], 'Type of detection model used - RESERVED.')
# floor button classification
flags.DEFINE_string('classification_floor_button_model_path', None,
                    'PATH to a SavedModel file capable of classification of elevator floor buttons.')
flags.DEFINE_enum('classification_floor_button_model_type', 'keras',
                  ['keras', 'reserved'], 'Type of classification model used - RESERVED.')
# ELEVATOR CONTROLS DETECTION FLAGS END
#####


FLAGS = flags.FLAGS
