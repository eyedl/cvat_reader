from typing import Tuple, Any, Optional

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

    def read_frame(self) -> Tuple[Optional[int], Any]:
        frame_id = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
        success, image = self.capture.read()
        if success:
            return frame_id, image
        else:
            return None, None

    def seek(self, frame_id: int):
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

    def close(self):
        if self.capture:
            self.capture.release()
            self.capture = None

    def get_number_of_frames(self) -> int:
        return int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
