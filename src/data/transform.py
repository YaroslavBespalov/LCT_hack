import torch
from torch import nn, Tensor



from albumentations.pytorch import ToTensor, ToTensorV2


from albumentations import *

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
                rotate_limit=10,  # rotate
                border_mode=cv2.BORDER_CONSTANT
            )
        ], p=0.4),
        # OneOf([
        #     # ElasticTransform(alpha=100, sigma=10, alpha_affine=1),
        #     # GridDistortion(),
        #     # OpticalDistortion(distort_limit=2, shift_limit=0.5),
        # ], p=0.5),
        post_transform()
    ])


def simple_transform(resize):
    return Compose([
        pre_transform(resize=resize),
        post_transform()
    ])
