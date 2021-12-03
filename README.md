# cvat_reader

Package to read cvat training set zip file into numpy array image and annotations.

The cvat format is usefull because it contains the original video file. The original video file has two main advantage over image files:
1. The original video file is much better compressed than a bunch of image files.
2. The image files are re-compressed versions of the video file and therefore lower in quality

## Install

```shell script
pip install cvat_reader
```

## Example

```python

import cv2

from cvat_reader import open_cvat


with open_cvat("training.zip") as dataset:
    print(dataset.labels)

    labels = {}
    for label in dataset.labels:
        h = label['color'].lstrip('#')
        labels[label['name']] = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

    for frame in dataset:
        if frame.annotations:
            img = frame.image.copy()

            for label in dataset.labels:
                for annotation in frame.annotations:
                    color = labels[annotation.label]

                    (x1, y1), (x2, y2) = annotation.bounding_box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color)

            cv2.imshow('image', img)
            cv2.waitKey(0)

```

By default the video is loaded and all image data is put in the `frame.image` attribute. When you are only interested in the data, or have an other way to process the video you can pass `load_video=False` to `open_cvat` and the images are not loaded. When you pass `load_video=False` this library does not depend on cv2 or numpy.


```python

from cvat_reader import open_cvat

def process_annotations(frame_id, annotations):
    ...

with open_cvat("training.zip", load_video=False) as dataset:
    print(dataset.labels)

    for frame in dataset:
        """
        >>> frame.image
        None
        """
        if frame.annotations:
            process_annotations(frame.frame_id, frame.annotations)
            

```

## Support

`cvat_reader` currently supports the following types of annotations:
- BoundingBox

Media types supported: *all types cv2 supports*


## Changelog

## 0.3.1 (2021-12-03)

Bugfix:
- Fix typing

## 0.3.0 (2021-12-03)

Feature:
- Add `occluded`, `outside` and `attributes` to `Annotation`. When `Annotation` is interpolated the `occluded`, `outside` and `attributes` fields are copied from the *first* non-interpolated annotation.

## 0.2.1 (2021-11-04)

Bugfix:
- Include last frame
- Include `video_file` in `Dataset`

## 0.2.0 (2021-11-04)

Feature:
- Add `load_video` flag to `open_cvat` to specify if video should be loaded too. This removes the dependecy on cv2/numpy when you don't need the video, or use another tool for processing.

### 0.1.2 (2021-10-26)

Bugfix:
- Stop iteration when last frame is reached

### 0.1.1 (2021-10-25)

Bugfix:
- data directory sometimes contains non-video files. Those files should not be picked as video files. This bugfix solves this by verifying if cv2 can load the file. 

### 0.1.0 (2021-10-22)

Feature:
- Properly read tracks and interpolate 