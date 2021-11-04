from typing import Tuple, Any

from .base import VideoReader


class DummyVideoReader(VideoReader):
    def __init__(self):
        self.frame_id = 0

    def read_frame(self) -> Tuple[int, Any]:
        frame_id = self.frame_id
        self.frame_id += 1
        return frame_id, None

    def seek(self, frame_id: int):
        self.frame_id = frame_id
