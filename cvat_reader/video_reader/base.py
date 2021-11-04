from abc import ABC, abstractmethod
from typing import Any, Tuple


class VideoReader(ABC):
    @abstractmethod
    def read_frame(self) -> Tuple[int, Any]:
        raise NotImplementedError

    @abstractmethod
    def seek(self, frame_id: int):
        raise NotImplementedError

    def close(self):
        pass
