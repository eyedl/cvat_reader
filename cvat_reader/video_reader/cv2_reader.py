from typing import Tuple, Any

try:
    import cv2
except ImportError as e:
    raise RuntimeError(
        "cv2 is required to use cvat_reader. You can install it using 'pip install opencv-python'"
    ) from e

from .base import VideoReader


class CV2Reader(VideoReader):
    def __init__(self, video_file: str):
        self.capture = cv2.VideoCapture(video_file)

    def read_frame(self) -> Tuple[int, Any]:
        frame_id = self.capture.get(cv2.cv2.CAP_PROP_POS_FRAMES)
        success, image = self.capture.read()
        return frame_id, image

    def seek(self, frame_id: int):
        self.capture.set(cv2.cv2.CAP_PROP_POS_FRAMES, frame_id)

    def close(self):
        if self.capture:
            self.capture.release()
            self.capture = None
