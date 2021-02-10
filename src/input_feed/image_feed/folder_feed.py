"""A class providing a synchronous folder image feed capability.
This module contains functions needed for reading an image from Folder.
"""
import acapture
import cv2

from src.utils.path_utils import force_separator_as_path_end
from src.utils.path_utils import check_path_existence
from src.input_feed.image_feed.ImageFeed import ImageFeed
from src.main.flags_global import FLAGS


class FolderFeedSync(ImageFeed):
    def __init__(self):
        self.cap = None
        self.image_dir_path = force_separator_as_path_end(FLAGS.image_input_folder_path)
        self.check = False
        self.frame = []

    def open_directory(self):
        if self.image_dir_path is None:
            raise ValueError("Path to the directory not specified, use .set_dir_path")
        if self.cap is None:
            self.cap = acapture.open(self.image_dir_path)  # recursive open

    def close_directory(self):
        self.cap = None
        self.image_dir_path = None

    def set_image_dir_path(self, image_dir_path):
        check_path_existence(image_dir_path, self.__name__)
        self.image_dir_path = force_separator_as_path_end(image_dir_path)

    def get_next_frame(self):
        self.open_directory()
        self.check, frame_raw = self.cap.read()
        if self.check:
            self.frame = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
        else:
            self.frame = None

    def get_frame(self):
        self.get_next_frame()
        return self.frame

    def show_frame(self, repeat=True):
        if repeat is True:
            wait_delay = 1
        else:
            wait_delay = 0
        while True:
            cv2.imshow("FolderFrame", self.frame)
            k = cv2.waitKey(wait_delay)
            if not repeat or k == 27:
                break
        cv2.destroyWindow("FolderFrame")
