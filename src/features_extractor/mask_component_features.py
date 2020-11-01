import itertools
from typing import List, Callable, Tuple

from scipy.spatial import distance
from scipy.spatial.distance import cdist
import torch
from torch import nn, Tensor
import numpy as np
import cv2 as cv
from scipy.ndimage import label, generate_binary_structure


def intersect_metric(component: np.ndarray, ref_mask: np.ndarray):
    norm = component.sum()
    return (component * ref_mask).sum() / norm, \
           norm / ref_mask.shape[-1]


def intersect_nearest(component: np.ndarray, ref_mask: np.ndarray):
    norm = component.sum()

    label_mask, num_features = label(ref_mask, structure=generate_binary_structure(2, 2))

    res = []

    for k in range(1, num_features + 1):
        mask_k = np.zeros_like(ref_mask, dtype=np.float32)
        mask_k[label_mask == k] = 1.0
        res.append((component * ref_mask).sum() / norm)

    return [max(res)] if len(res) > 0 else [0]


def get_contours(mask: np.ndarray, min_len: int = 0):

        mask[mask > 0.5] = 1
        mask[mask < 0.5] = 0
        mask = mask.astype(np.uint8)

        contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        filtered = [c for c in contours if c.shape[0] >= min_len]

        if len(filtered) == 0:
            return None

        union = np.concatenate(filtered)
        union = union.astype(np.float32)

        n = union.shape[0]
        if n > 100:
            idx = np.linspace(n - 1, 0, 100).astype(int)
            return union[idx, 0, :] / mask.shape[-1]
        else:
            return union[:, 0, :] / mask.shape[-1]


def contour_min_distance_metric(component: np.ndarray, ref_mask: np.ndarray):

    if (component * ref_mask).sum() > 1:
        return [0]

    cont = get_contours(component)
    ref_cont = get_contours(ref_mask, 10)

    if cont is None or ref_cont is None:
        return [0]

    d = cdist(cont, ref_cont, 'euclidean').reshape(-1)

    return [float(np.min(d))]


def coordinates_metric(component: np.ndarray, ref_mask: np.ndarray):

    indices = torch.nonzero(torch.from_numpy(component > 0.5)) / float(ref_mask.shape[-1])
    indices = indices[:, [1, 0]]
    center = indices.mean(0)
    return center[0].item(), center[1].item()


class ComponentFeaturesExtractor:

    def __init__(self,
                 metrics: List[ Callable[[np.ndarray, np.ndarray], Tuple[float]] ],
                 aggregators: List[Callable[[np.ndarray, int], np.ndarray]]
                 ):
        self.metrics = metrics
        self.bs = generate_binary_structure(2, 2)
        self.aggregators = aggregators

    def extract_for_one_component(self, component: np.ndarray, ref_mask: np.ndarray) -> List[float]:
        return list(itertools.chain.from_iterable(metric(component, ref_mask) for metric in self.metrics))

    def extract(self, mask: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:

        label_mask, num_features = label(mask, structure=self.bs)

        res = []

        for k in range(1, num_features + 1):
            mask_k = np.zeros_like(mask, dtype=np.float32)
            mask_k[label_mask == k] = 1.0
            fich = np.array(self.extract_for_one_component(mask_k, ref_mask))
            res.append(fich.reshape(1, -1))

        if len(res) == 0:
            return None

        cat = np.concatenate(res)

        return np.concatenate([agg(cat, 0) for agg in self.aggregators])

