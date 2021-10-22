# cvat_reader

Package to read cvat training sets into numpy array (image) and annotations.

The cvat format is usefull because it contains the original video file. The original video file has two main advantage over image files:
1. The original video file is much better compressed than a bunch of image files.
2. The image files are re-compressed versions of the video file and therefore lower in quality

## Example

```python

import cv2

from cvat_reader import open_cvat


with open_cvat("task.zip") as dataset:
    dataset.seek(100)
    print(dataset.labels)
    dataset.seek_first_annotation()

    for frame in dataset:
        if frame.annotations:
            cv2.imshow('original', frame.image)

```
