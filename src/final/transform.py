import cv2

from albumentations.pytorch import ToTensor
from albumentations import (
    Resize,
    Compose,
    Normalize,
    OneOf,
    ShiftScaleRotate
)


def pre_transform(resize):
    transforms = []
    transforms.append(Resize(resize, resize))
    return Compose(transforms)


def post_transform():
    return Compose([
        Normalize(
            mean=(0.485),
            std=(0.5)),
        ToTensor()
    ])


def mix_transform(resize):
    return Compose([
        pre_transform(resize=resize),
        OneOf([
            ShiftScaleRotate(
                shift_limit=0,
                scale_limit=0.1,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT
            )
        ], p=0.4),
        post_transform()
    ])


def simple_transform(resize):
    return Compose([
        pre_transform(resize=resize),
        post_transform()
    ])
