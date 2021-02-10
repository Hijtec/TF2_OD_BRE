"""A class providing a video image feed capability.
This module contains functions needed for reading an image from VideoDevice.
"""
import acapture
import cv2

from src.utils.path_utils import check_path_existence
from src.input_feed.image_feed.ImageFeed import ImageFeed


class VideoFeedAsync(ImageFeed):
    def __init__(self, video_file_path=None, frame_capture=False):
        self.cap = None
        self.video_file_path = video_file_path
        self.check = False
        self.frame = []
        self.frame_capture = frame_capture

    def open_video(self):
        if self.cap is None:
            self.cap = acapture.open(self.video_file_path, frame_capture=self.frame_capture)

    def close_video(self):
        self.cap.destroy()
        self.cap = None

    def set_video_file_path(self, video_file_path):
        check_path_existence(video_file_path, self.__name__)
        self.video_file_path = video_file_path

    def get_frame(self):
        self.get_next_frame()
        return self.frame

    def get_next_frame(self):
        self.open_video()
        self.check, frame_raw = self.cap.read()
        if self.check:
            self.frame = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
        else:
            self.frame = None

    def start_video_stream(self):
        while self.frame is not None:
            self.get_next_frame()

    def show_frame(self, repeat=True):
        if repeat is True:
            wait_delay = 1
        else:
            wait_delay = 0
        while True:
            cv2.imshow("VideoFrame", self.frame)
            k = cv2.waitKey(wait_delay)
            if not repeat or k == 27:
                break
        cv2.destroyWindow("VideoFrame")
