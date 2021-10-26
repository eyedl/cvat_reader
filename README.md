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


## Support

`cvat_reader` currently supports the following types of annotations:
- BoundingBox

Media types supported: *all types cv2 supports*


## Changelog

### 0.1.2 (2021-10-26)

Bugfix:
- Stop iteration when last frame is reached

### 0.1.1 (2021-10-25)

Bugfix:
- data directory sometimes contains non-video files. Those files should not be picked as video files. This bugfix solves this by verifying if cv2 can load the file. 

### 0.1.0 (2021-10-22)

Feature:
- Properly read tracks and interpolate 