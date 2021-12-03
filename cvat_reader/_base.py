import json
import glob

import logging
import os
import shutil
import tempfile
import zipfile
from contextlib import contextmanager

from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, ContextManager, Iterable

from cvat_reader.video_reader.base import VideoReader

logger = logging.getLogger(__name__)


@dataclass
class Annotation:
    frame_id: int
    track_id: int
    label: str
    bounding_box: Tuple[Tuple[int, int], Tuple[int, int]]
    occluded: bool
    outside: bool
    attributes: dict
    interpolated: bool

    @classmethod
    def from_interpolation(
        cls, begin_annotation: "Annotation", end_annotation: "Annotation", frame_id: int
    ) -> "Annotation":
        relative_offset: float = (frame_id - begin_annotation.frame_id) / (
            end_annotation.frame_id - begin_annotation.frame_id
        )

        return cls(
            frame_id=frame_id,
            track_id=begin_annotation.track_id,
            label=begin_annotation.label,
            occluded=begin_annotation.occluded,
            outside=begin_annotation.outside,
            attributes=begin_annotation.attributes,
            interpolated=True,
            bounding_box=(
                (
                    int(
                        (
                            end_annotation.bounding_box[0][0]
                            - begin_annotation.bounding_box[0][0]
                        )
                        * relative_offset
                        + begin_annotation.bounding_box[0][0]
                    ),
                    int(
                        (
                            end_annotation.bounding_box[0][1]
                            - begin_annotation.bounding_box[0][1]
                        )
                        * relative_offset
                        + begin_annotation.bounding_box[0][1]
                    ),
                ),
                (
                    int(
                        (
                            end_annotation.bounding_box[1][0]
                            - begin_annotation.bounding_box[1][0]
                        )
                        * relative_offset
                        + begin_annotation.bounding_box[1][0]
                    ),
                    int(
                        (
                            end_annotation.bounding_box[1][1]
                            - begin_annotation.bounding_box[1][1]
                        )
                        * relative_offset
                        + begin_annotation.bounding_box[1][1]
                    ),
                ),
            ),
        )


@dataclass
class Track:
    track_id: int
    annotations: List[Annotation]

    @property
    def first_frame_id(self) -> int:
        return self.annotations[0].frame_id

    @property
    def last_frame_id(self) -> int:
        return self.annotations[-1].frame_id

    @property
    def label(self) -> str:
        return self.annotations[0].label

    def get_annotation(self, frame_id: int) -> Optional[Annotation]:
        if frame_id < self.first_frame_id:
            return None

        if frame_id > self.last_frame_id:
            return None

        for i, annotation in enumerate(self.annotations):
            if annotation.frame_id == frame_id:
                return annotation

            # i CANNOT be 0 due to previous checks
            if annotation.frame_id > frame_id:
                prev_annotation = self.annotations[i - 1]
                return Annotation.from_interpolation(
                    begin_annotation=prev_annotation,
                    end_annotation=annotation,
                    frame_id=frame_id,
                )

        # This shouldn't happen
        return None


@dataclass
class Frame:
    frame_id: int
    image: Any
    annotations: List[Annotation]


@contextmanager
def temp_dir():
    dir_path = tempfile.mkdtemp()
    try:
        yield dir_path
    finally:
        shutil.rmtree(dir_path)


class Dataset(Iterable[Frame]):
    def __init__(
        self, task_file: str, annotations_file: str, video_file: str, load_video: bool
    ):
        with open(task_file, "r") as fp:
            json_data = json.load(fp)
            self.labels: List[dict] = json_data["labels"]

        with open(annotations_file, "r") as fp:
            json_data = json.load(fp)

            self.tracks: List[Track] = []

            for track_id, track in enumerate(json_data[0]["tracks"]):
                annotations = []
                for shape in track["shapes"]:
                    frame_id = shape["frame"]

                    if shape["type"] == "rectangle":
                        x1, y1, x2, y2 = shape["points"]
                        annotation = Annotation(
                            frame_id=frame_id,
                            track_id=track_id,
                            label=track["label"],
                            bounding_box=((int(x1), int(y1)), (int(x2), int(y2))),
                            interpolated=False,
                            occluded=shape["occluded"],
                            outside=shape["outside"],
                            attributes=shape["attributes"],
                        )
                        annotations.append(annotation)
                    else:
                        logger.debug(f"Skipping annotation of type {shape['type']}")

                if annotations:
                    self.tracks.append(
                        Track(track_id=track_id, annotations=annotations)
                    )

        if load_video:
            from .video_reader.cv2_reader import CV2Reader

            self.video_reader: VideoReader = CV2Reader(video_file)
        else:
            from .video_reader.dummy_reader import DummyVideoReader

            self.video_reader = DummyVideoReader()

        self.video_file = video_file
        self.last_frame_id = max(track.last_frame_id for track in self.tracks)

    def seek(self, frame_id: int):
        self.video_reader.seek(frame_id)

    def seek_first_annotation(self):
        first_frame_id = min(track.first_frame_id for track in self.tracks)
        self.seek(first_frame_id)

    def __iter__(self) -> "Dataset":
        return self

    def __next__(self) -> Frame:
        frame_id, image = self.video_reader.read_frame()
        if frame_id > self.last_frame_id:
            raise StopIteration()

        annotations = [track.get_annotation(frame_id) for track in self.tracks if track]

        return Frame(
            frame_id=frame_id,
            image=image,
            annotations=[annotation for annotation in annotations if annotation],
        )

    def close(self):
        self.video_reader.close()

    def __del__(self):
        self.close()


def is_video_file(filepath: str) -> bool:
    return os.path.splitext(filepath)[1] in (".mp4", ".mpeg", ".mov", ".avi")


def open_cvat(filename: str, load_video: bool = True) -> ContextManager[Dataset]:
    """
    This code uses a ugly hack described here:
    https://stackoverflow.com/questions/49335263/how-to-properly-annotate-a-contextmanager-in-pycharm#answer-56611439
    The hack is needed because PyCharm isn't able to infer the types of a contextmanager correctly. See
    bug report: https://youtrack.jetbrains.com/issue/PY-36444
    """

    @contextmanager
    def _open(filename, load_video):
        with temp_dir() as dir_path:
            logger.debug(f"Extracting {filename} to {dir_path}")
            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(dir_path)

            video_files = [
                filename
                for filename in glob.glob(f"{dir_path}/data/*")
                if is_video_file(filename)
            ]
            if not video_files:
                raise Exception("There are no video files")

            video_file = video_files[0]
            logger.debug(f"Going to read video frame {video_file}")

            dataset = Dataset(
                task_file=f"{dir_path}/task.json",
                annotations_file=f"{dir_path}/annotations.json",
                video_file=video_file,
                load_video=load_video,
            )

            try:
                yield dataset
            finally:
                dataset.close()

            logger.debug("Cleaning temp directory")

    return _open(filename, load_video)


__all__ = ["open_cvat", "Annotation", "Dataset", "Frame"]
