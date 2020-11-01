import os
import random
from typing import Dict

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.transforms import ToTensor

from data.mask_component_features import coordinates_metric, ComponentFeaturesExtractor, contour_min_distance_metric, \
    intersect_metric
from data.path import Paths
import numpy as np
from PIL import Image

from data.transform import mix_transform


class Figure:

    CIRCLE = "circle"
    RECT = "rectangle"

    def __init__(self, df: pd.Series):
        df = df.to_dict()
        self.xc = float(df[" xcenter"])
        self.yc = float(df[" ycenter"])
        self.rh = float(df[" rhorizontal"])
        self.rv = float(df[" rvertical"])
        self.shape = str(df[' shape']).strip()
        assert self.shape in [Figure.CIRCLE, Figure.RECT]


class ExpertAlgo(Dataset):

    def __init__(self, path="../../data/Dataset", transforms=mix_transform(256)):
        self.scores_csv = pd.read_csv(path + "/OpenPart.csv", names=['Case', 'Sample 1', 'Sample 2', 'Sample 3'], skiprows=1)
        self.path = path
        self.img_files = os.listdir(path + "/Origin")
        self.figures = pd.read_csv(self.path + "/DX_TEST_RESULT_FULL.csv")
        self.iou_fich = pd.read_csv(self.path + "/../iou_features_v01.csv")
        self.iou_keys = ['overall_iou', 'overall_instance_iou', 'back_iou',
       'left_lung_iou', 'right_lung_iou', 'back_instance_iou',
       'left_lung_inst_iou', 'right_lung_inst_iou']

        # print(self.iou_fich.keys())
        self.transforms = transforms

        self.saved_masks = dict()
        self.saved_images = dict()

    def __len__(self):
        return self.scores_csv.shape[0]

    def __getitem__(self, index) -> Dict[str, Tensor]:

        if index not in self.saved_images:
            image_path = self.path + "/Origin/" + self.scores_csv.iloc[index]["Case"]
            image = np.array(Image.open(image_path).convert("RGB"))
            self.saved_images[index] = image

        image = self.saved_images[index]

        tmp = self.scores_csv.iloc[index]["Case"].split(".")[0]

        if index not in self.saved_masks:
            expert = np.array(Image.open(os.path.join(self.path, "Expert", f"{tmp}_expert.png")), dtype=np.float32)
            sample_1 = np.array(Image.open(os.path.join(self.path, "sample_1", f"{tmp}_s1.png")), dtype=np.float32)
            sample_2 = np.array(Image.open(os.path.join(self.path, "sample_2", f"{tmp}_s2.png")), dtype=np.float32)
            sample_3 = np.array(Image.open(os.path.join(self.path, "sample_3", f"{tmp}_s3.png")), dtype=np.float32)
            self.saved_masks[index] = {
                "Expert": expert,
                "sample_1": sample_1,
                "sample_2": sample_2,
                "sample_3": sample_3,
            }

        expert = self.saved_masks[index]["Expert"]
        sample_1 = self.saved_masks[index]["sample_1"]
        sample_2 = self.saved_masks[index]["sample_2"]
        sample_3 = self.saved_masks[index]["sample_3"]

        # figures_tmp = self.figures.loc[self.figures['file_name'] == tmp]

        # sample_1_figures = figures_tmp.loc[figures_tmp[' user_name'] == "sample_1"]
        # sample_1_figures = [Figure(df) for index, df in sample_1_figures.iterrows()]
        # sample_2_figures = figures_tmp.loc[figures_tmp[' user_name'] == "sample_2"]
        # sample_2_figures = [Figure(df) for index, df in sample_2_figures.iterrows()]
        # sample_3_figures = figures_tmp.loc[figures_tmp[' user_name'] == "sample_3"]
        # sample_3_figures = [Figure(df) for index, df in sample_3_figures.iterrows()]
        # expert_figures = figures_tmp.loc[figures_tmp[' user_name'] == "Expert"]
        # expert_figures = [Figure(df) for index, df in expert_figures.iterrows()]

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

        dict_image = self.transforms(image=image, masks=all_masks)
        res_image = dict_image["image"]
        res_mask = dict_image["masks"]

        label_1 = self.scores_csv.iloc[index]['Sample 1']
        label_2 = self.scores_csv.iloc[index]['Sample 2']
        label_3 = self.scores_csv.iloc[index]['Sample 3']

        tot = ToTensor()

        return {
            "image": res_image,
            "Expert": tot(res_mask[0]),
            "sample_1": tot(res_mask[1]),
            "sample_2": tot(res_mask[2]),
            "sample_3": tot(res_mask[3]),
            "label_1": torch.tensor(int(label_1)),
            "label_2": torch.tensor(int(label_2)),
            "label_3": torch.tensor(int(label_3)),
            "iou_1": np.array(iou_1),
            "iou_2": np.array(iou_2),
            "iou_3": np.array(iou_3)
        }


class ExpertAlgoTest(Dataset):

    def __init__(self, path="../../data/Dataset", transforms=mix_transform(256)):
        self.path = path
        self.img_files = os.listdir(path + "/Origin")
        # self.figures = pd.read_csv(self.path + "/DX_TEST_RESULT_FULL.csv")
        self.iou_fich = pd.read_csv(self.path + "/../iou_features_v01.csv")
        self.iou_keys = ['overall_iou', 'overall_instance_iou', 'back_iou',
       'left_lung_iou', 'right_lung_iou', 'back_instance_iou',
       'left_lung_inst_iou', 'right_lung_inst_iou']

        self.test_keys = pd.read_csv(path + "/TestPart.csv", names=['Case'], skiprows=1)

        # print(self.iou_fich.keys())
        self.transforms = transforms

        self.saved_masks = dict()
        self.saved_images = dict()

    def __len__(self):
        return self.test_keys.shape[0]

    def __getitem__(self, index) -> Dict[str, Tensor]:

        if index not in self.saved_images:
            image_path = self.path + "/Origin/" + self.test_keys.iloc[index]["Case"]
            image = np.array(Image.open(image_path).convert("RGB"))
            self.saved_images[index] = image

        image = self.saved_images[index]

        tmp = self.test_keys.iloc[index]["Case"].split(".")[0]

        if index not in self.saved_masks:
            expert = np.array(Image.open(os.path.join(self.path, "Expert", f"{tmp}_expert.png")), dtype=np.float32)
            sample_1 = np.array(Image.open(os.path.join(self.path, "sample_1", f"{tmp}_s1.png")), dtype=np.float32)
            sample_2 = np.array(Image.open(os.path.join(self.path, "sample_2", f"{tmp}_s2.png")), dtype=np.float32)
            sample_3 = np.array(Image.open(os.path.join(self.path, "sample_3", f"{tmp}_s3.png")), dtype=np.float32)
            self.saved_masks[index] = {
                "Expert": expert,
                "sample_1": sample_1,
                "sample_2": sample_2,
                "sample_3": sample_3,
            }

        expert = self.saved_masks[index]["Expert"]
        sample_1 = self.saved_masks[index]["sample_1"]
        sample_2 = self.saved_masks[index]["sample_2"]
        sample_3 = self.saved_masks[index]["sample_3"]

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

        dict_image = self.transforms(image=image, masks=all_masks)
        res_image = dict_image["image"]
        res_mask = dict_image["masks"]

        tot = ToTensor()

        return {
            "image": res_image,
            "Expert": tot(res_mask[0]),
            "sample_1": tot(res_mask[1]),
            "sample_2": tot(res_mask[2]),
            "sample_3": tot(res_mask[3]),
            "iou_1": np.array(iou_1),
            "iou_2": np.array(iou_2),
            "iou_3": np.array(iou_3)
        }


if __name__ == "__main__":

    dataset = ExpertAlgoTest(transforms=mix_transform(256))

    print(dataset[1])









