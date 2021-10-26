import json
import glob

import logging
import shutil
import tempfile
import zipfile
from contextlib import contextmanager

from dataclasses import dataclass
from typing import List, Tuple, Iterator, Optional

try:
    import numpy
except ImportError as e:
    raise RuntimeError("Numpy is required to use cvat_reader. You can install it using 'pip install numpy'") from e

try:
    import cv2
except ImportError as e:
    raise RuntimeError("cv2 is required to use cvat_reader. You can install it using 'pip install opencv-python'") from e


logger = logging.getLogger(__name__)


@dataclass
class Annotation:
    frame_id: int
    track_id: int
    label: str
    bounding_box: Tuple[Tuple[int, int], Tuple[int, int]]
    interpolated: bool

    @classmethod
    def from_interpolation(cls, begin_annotation: 'Annotation', end_annotation: 'Annotation', frame_id: int) -> 'Annotation':
        relative_offset: float = (frame_id - begin_annotation.frame_id) / (end_annotation.frame_id - begin_annotation.frame_id)

        return cls(
            frame_id=frame_id,
            track_id=begin_annotation.track_id,
            label=begin_annotation.label,
            interpolated=True,
            bounding_box=(
                (
                        int((end_annotation.bounding_box[0][0] - begin_annotation.bounding_box[0][0]) * relative_offset
                        + begin_annotation.bounding_box[0][0]),
                        int((end_annotation.bounding_box[0][1] - begin_annotation.bounding_box[0][1]) * relative_offset
                        + begin_annotation.bounding_box[0][1])
                ),
                (
                        int((end_annotation.bounding_box[1][0] - begin_annotation.bounding_box[1][0]) * relative_offset
                        + begin_annotation.bounding_box[1][0]),

                        int((end_annotation.bounding_box[1][1] - begin_annotation.bounding_box[1][1]) * relative_offset
                        + begin_annotation.bounding_box[1][1])
                )
            )
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
                    frame_id=frame_id
                )

        # This shouldn't happen
        return None


@dataclass
class Frame:
    frame_id: int
    image: numpy.array
    annotations: List[Annotation]


@contextmanager
def temp_dir():
    dir_path = tempfile.mkdtemp()
    try:
        yield dir_path
    finally:
        shutil.rmtree(dir_path)


class Dataset:
    def __init__(self, task_file: str, annotations_file: str, video_file: str):
        with open(task_file, 'r') as fp:
            json_data = json.load(fp)
            self.labels: List[dict] = json_data['labels']

        with open(annotations_file, 'r') as fp:
            json_data = json.load(fp)

            self.tracks: List[Track] = []

            for track_id, track in enumerate(json_data[0]['tracks']):
                annotations = []
                for shape in track['shapes']:
                    frame_id = shape['frame']

                    if shape['type'] == 'rectangle':
                        x1, y1, x2, y2 = shape['points']
                        annotation = Annotation(
                            frame_id=frame_id,
                            track_id=track_id,
                            label=track['label'],
                            bounding_box=(
                                (int(x1), int(y1)), (int(x2), int(y2))
                            ),
                            interpolated=False
                        )
                        annotations.append(annotation)
                    else:
                        logger.debug(f"Skipping annotation of type {shape['type']}")

                if annotations:
                    self.tracks.append(
                        Track(
                            track_id=track_id,
                            annotations=annotations
                        )
                    )

        self.capture = cv2.VideoCapture(video_file)
        self.frame_count = int(self.capture.get(cv2.cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.capture.get(cv2.cv2.CAP_PROP_FPS)
        self.current_frame = 0

    def seek(self, frame_id: int):
        self.capture.set(cv2.cv2.CAP_PROP_POS_FRAMES, frame_id)
        self.current_frame = frame_id

    def seek_first_annotation(self):
        first_frame_id = min(
            track.first_frame_id for track in self.tracks
        )
        self.seek(first_frame_id)

    def __iter__(self) -> Iterator[Frame]:
        return self

    def __next__(self) -> Frame:
        self.current_frame = frame_id = self.capture.get(cv2.cv2.CAP_PROP_POS_FRAMES)
        if self.current_frame >= self.frame_count:
            raise StopIteration()

        success, image = self.capture.read()

        annotations = [
            track.get_annotation(frame_id)
            for track in self.tracks
            if track
        ]

        return Frame(
            frame_id=frame_id,
            image=image,
            annotations=[annotation for annotation in annotations if annotation]
        )

    def close(self):
        if self.capture:
            logger.debug("Closing dataset")
            self.capture.release()
            self.capture = None

    def __del__(self):
        self.close()


def cv2_can_read(filepath: str) -> bool:
    capture = cv2.VideoCapture(filepath)
    can_read = capture.isOpened()
    capture.release()
    return can_read


@contextmanager
def open_cvat(filename: str) -> Dataset:
    with temp_dir() as dir_path:
        logger.debug(f"Extracting {filename} to {dir_path}")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(dir_path)

        video_files = [
            filename for filename in glob.glob(f"{dir_path}/data/*")
            if cv2_can_read(filename)
        ]
        if not video_files:
            raise Exception("There are no video files")

        video_file = video_files[0]
        logger.debug(f"Going to read video frame {video_file}")

        dataset = Dataset(
            task_file=f"{dir_path}/task.json",
            annotations_file=f"{dir_path}/annotations.json",
            video_file=video_file
        )

        try:
            yield dataset
        finally:
            dataset.close()

        logger.debug("Cleaning temp directory")


