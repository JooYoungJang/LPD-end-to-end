from .car_carplate_two_stage_end2end import CAR_CARPLATE_TWO_STAGE_END2ENDDetection, CAR_CARPLATE_TWO_STAGE_END2ENDAnnotationTransform, CAR_CARPLATE_TWO_STAGE_END2END_CLASSES
from .car_carplate import CAR_CARPLATEDetection, CAR_CARPLATEAnnotationTransform, CAR_CARPLATE_CLASSES, CAR_CARPLATE_ROOT, CARPLATE_CLASSES

from .config import *
import torch
import cv2
import numpy as np

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def base_transform(image, size, mean):
    if type(size) == type(()):
        x = cv2.resize(image, (size[1], size[0])).astype(np.float32)
    else:
        x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x

class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels
