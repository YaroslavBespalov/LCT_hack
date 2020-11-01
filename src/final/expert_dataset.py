import os
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from transform import mix_transform


class Figure:

    CIRCLE = 'circle'
    RECT = 'rectangle'

    def __init__(self, df: pd.Series):
        df = df.to_dict()
        self.xc = float(df[' xcenter'])
        self.yc = float(df[' ycenter'])
        self.rh = float(df[' rhorizontal'])
        self.rv = float(df[' rvertical'])
        self.shape = str(df[' shape']).strip()
        assert self.shape in [Figure.CIRCLE, Figure.RECT]


class ExpertDataset(Dataset):

    def __init__(self, path, enable_transforms=False):
        self.scores_csv = pd.read_csv(
            os.path.join(path, 'Dataset', 'OpenPart.csv'),
            names=['Case', 'Sample 1', 'Sample 2', 'Sample 3'],
            skiprows=1
        )
        self.path = path
        self.img_files = os.listdir(os.path.join(path, 'Dataset', 'Origin'))
        self.figures = pd.read_csv(os.path.join(path, 'Dataset', 'DX_TEST_RESULT_FULL.csv'))
        self.iou_fich = pd.read_csv(os.path.join(path, 'iou_features_v01.csv'))
        self.iou_keys = [
            'overall_iou', 'overall_instance_iou',
            'back_iou', 'left_lung_iou', 'right_lung_iou',
            'back_instance_iou', 'left_lung_inst_iou', 'right_lung_inst_iou'
        ]
        self.enable_transforms = enable_transforms
        self.transforms = mix_transform(256)

    def __len__(self):
        return self.scores_csv.shape[0]

    def __getitem__(self, index) -> Dict[str, Tensor]:
        image_path = os.path.join(self.path, 'Dataset', 'Origin', self.scores_csv.iloc[index]['Case'])

        image = np.array(Image.open(image_path).convert('RGB'))
        tmp = self.scores_csv.iloc[index]['Case'].split('.')[0]

        expert = np.array(Image.open(os.path.join(self.path, 'Dataset', 'Expert', f'{tmp}_expert.png')), dtype=np.float32)
        sample_1 = np.array(Image.open(os.path.join(self.path, 'Dataset', 'sample_1', f'{tmp}_s1.png')), dtype=np.float32)
        sample_2 = np.array(Image.open(os.path.join(self.path, 'Dataset', 'sample_2', f'{tmp}_s2.png')), dtype=np.float32)
        sample_3 = np.array(Image.open(os.path.join(self.path, 'Dataset', 'sample_3', f'{tmp}_s3.png')), dtype=np.float32)

        iou_tmp = self.iou_fich.loc[self.iou_fich['key'] == tmp]
        iou_1 = iou_tmp[iou_tmp['sample_i'] == 0]
        iou_1 = [iou_1.iloc[0][k] for k in self.iou_keys]
        iou_2 = iou_tmp[iou_tmp['sample_i'] == 1]
        iou_2 = [iou_2.iloc[0][k] for k in self.iou_keys]
        iou_3 = iou_tmp[iou_tmp['sample_i'] == 2]
        iou_3 = [iou_3.iloc[0][k] for k in self.iou_keys]

        all_masks = [
            expert,
            sample_1,
            sample_2,
            sample_3
        ]

        if self.enable_transforms:
            dict_image = self.transforms(image=image, masks=all_masks)
            res_image = dict_image['image']
            res_mask = dict_image['masks']
        else:
            res_image, res_mask = image, all_masks

        label_1 = self.scores_csv.iloc[index]['Sample 1']
        label_2 = self.scores_csv.iloc[index]['Sample 2']
        label_3 = self.scores_csv.iloc[index]['Sample 3']

        tot = ToTensor()

        return {
            'image': res_image,
            'Expert': tot(res_mask[0]),
            'sample_1': tot(res_mask[1]),
            'sample_2': tot(res_mask[2]),
            'sample_3': tot(res_mask[3]),
            'label_1': torch.tensor(int(label_1)),
            'label_2': torch.tensor(int(label_2)),
            'label_3': torch.tensor(int(label_3)),
            'iou_1': np.array(iou_1),
            'iou_2': np.array(iou_2),
            'iou_3': np.array(iou_3)
        }
