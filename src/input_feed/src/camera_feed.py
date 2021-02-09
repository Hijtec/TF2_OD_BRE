"""A script providing an asynchronous camera image feed capability.
This module contains functions needed for reading an image from VideoDevice.
"""
import acapture
import cv2


class CameraFeedAsync:
    def __init__(self, cam_src=0):
        self.cap = None
        self.check = False
        self.frame = []
        self.src = cam_src

    def open_camera_device(self):
        if self.cap is None:
            self.cap = acapture.open(self.src)  # /dev/video0 on linux

    def close_camera_device(self):
        self.cap.destroy()
        self.cap = None

    def get_next_frame(self):
        self.open_camera_device()
        self.check, frame_raw = self.cap.read()
        if self.check:
            self.frame = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
        else:
            self.frame = None

    def start_camera_stream(self):
        while self.frame is not None:
            self.get_next_frame()  # non-blocking

    def get_frame(self):
        self.get_next_frame()
        return self.frame

    def show_frame(self, repeat=True):
        if repeat is True:
            wait_delay = 1
        else:
            wait_delay = 0
        while True:
            cv2.imshow("CameraFrame", self.frame)
            k = cv2.waitKey(wait_delay)
            if not repeat or k == 27:
                break
        cv2.destroyWindow("CameraFrame")
