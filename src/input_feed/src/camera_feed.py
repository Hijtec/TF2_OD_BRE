import acapture
import cv2


class CameraFeed:
    def __init__(self, cam_src=0):
        self.cap = None
        self.check = False
        self.frame = []
        self.src = cam_src

    def open_camera_device(self):
        if self.cap is None:
            self.cap = acapture.open(self.src)  # /dev/video0 on linux

    def capture_frame(self):
        self.open_camera_device()
        self.check, frame_raw = self.cap.read()
        if self.check:
            self.frame = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
            return self.frame
        else:
            self.frame = None
            return None

    def start_camera_stream(self):
        self.open_camera_device()
        while True:
            self.capture_frame()  # non-blocking

    def get_frame(self):
        return self.frame

    def show_frame(self):
        cv2.imshow("CameraFrame", self.frame)
        cv2.waitKey(0)
        cv2.destroyWindow("CameraFrame")
