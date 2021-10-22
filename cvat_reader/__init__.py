import json
import glob

import logging
import shutil
import tempfile
import zipfile
from collections import defaultdict
from contextlib import contextmanager

from dataclasses import dataclass
from typing import List, Tuple, Union, Iterator, Dict

try:
    import numpy
except ImportError as e:
    raise RuntimeError("Numpy is required to use cvat_reader") from e

try:
    import cv2
except ImportError as e:
    raise RuntimeError("cv2 is required to use cvat_reader") from e


logger = logging.getLogger(__name__)


@dataclass
class PointAnnotation:
    label: str
    point: Tuple[int, int]


@dataclass
class BBoxAnnotation:
    label: str
    bounding_box: Tuple[Tuple[int, int], Tuple[int, int]]


@dataclass
class Frame:
    frame_id: int
    image: numpy.array
    annotations: Dict[str, List[Union[BBoxAnnotation, PointAnnotation]]]


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

            self.annotations_per_frame = defaultdict(lambda:defaultdict(list))

            for track in json_data[0]['tracks']:
                for shape in track['shapes']:
                    frame_id = shape['frame']

                    if shape['type'] == 'points':
                        x, y = shape['points']
                        annotation = PointAnnotation(
                            label=track['label'],
                            point=(int(x), int(y))
                        )
                    elif shape['type'] == 'rectangle':
                        x1, y1, x2, y2 = shape['points']
                        annotation = BBoxAnnotation(
                            label=track['label'],
                            bounding_box=(
                                (int(x1), int(y1)), (int(x2), int(y2))
                            )
                        )
                    else:
                        logger.debug(f"Skipping annotation of type {shape['type']}")

                    self.annotations_per_frame[frame_id][annotation.label].append(annotation)

        self.capture = cv2.VideoCapture(video_file)
        self.frame_count = int(self.capture.get(cv2.cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.capture.get(cv2.cv2.CAP_PROP_FPS)
        self.current_frame_id = 0

    def seek(self, frame_id: int):
        self.current_frame_id = frame_id
        self.capture.set(cv2.cv2.CAP_PROP_POS_FRAMES, self.current_frame_id)

    def seek_first_annotation(self):
        self.seek(min(self.annotations_per_frame.keys()))

    def __iter__(self) -> Iterator[Frame]:
        return self

    def __next__(self) -> Frame:
        frame_id = self.current_frame_id
        success, image = self.capture.read()
        self.current_frame_id += 1

        return Frame(
            frame_id=frame_id,
            image=image,
            annotations=self.annotations_per_frame.get(frame_id, [])
        )

    def close(self):
        if self.capture:
            logger.debug("Closing dataset")
            self.capture.release()
            self.capture = None

    def __del__(self):
        self.close()


@contextmanager
def open_cvat(filename: str) -> Dataset:
    with temp_dir() as dir_path:
        logger.debug(f"Extracting {filename} to {dir_path}")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(dir_path)
        logger.info("Done")

        video_files = glob.glob(f"{dir_path}/data/*")
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


