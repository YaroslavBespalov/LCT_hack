
import skimage
import numpy as np
from PIL import Image
from albumentations import Resize

from data.transform import simple_transform


def get_iou(x1, x2):
    intersection = np.clip((x1 * x2), 0, 1).sum()
    union = np.clip((x1 + x2), 0, 1).sum()

    if intersection == 0 and union == 0:
        iou = 1.
    elif union == 0:
        iou = 0.
    else:
        iou = intersection / union
    return iou

def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn

def get_instance_iou(gt, pr, thresholds, beta=1, verbose=1):
    """
    Calculate instance-wise F-score in range(0.5, 1, 0.05)

    Source:
        https://github.com/selimsef/dsb2018_topcoders/blob/master/selim/metric.py

    Args:
        gt: ground truth instances mask (each instance has its own unique value)
        pr: predicted instances mask (each instance has its own unique value)
        beta: F-score beta coeffitient
        verbose: verbosity level

    Returns:
        score: float
    """

    # separate instances
    gt = skimage.measure.label(gt)
    pr = skimage.measure.label(pr)

    print_fn = lambda x: print(x) if verbose else None

    true_objects = len(np.unique(gt))
    pred_objects = len(np.unique(pr))

    # Compute intersection between all objects
    intersection = np.histogram2d(gt.flatten(), pr.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(gt, bins=true_objects)[0]
    area_pred = np.histogram(pr, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Loop over IoU thresholds
    prec = []
    print_fn("Thresh\tTP\tFP\tFN\tPrec.")
    for t in thresholds:
        tp, fp, fn = precision_at(t, iou)
        if tp + fp + fn == 0:
            p = 1.
        else:
            p = (1 + beta ** 2) * tp / ((1 + beta ** 2) * tp + fp + beta ** 2 * fn + 1e-10)
        print_fn("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    print_fn("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))

    return prec


def split_mask_to_zones(mask, lungs_labeled_mask, intersection_threshold=0.3):
    # split lungs mask to separate masks, one area for each mask
    lung_labels = [0, 1, 2]
    assert np.unique(lungs_labeled_mask).tolist() == lung_labels

    lung_areas = [lungs_labeled_mask == label for label in lung_labels]

    # placeholder for result
    background_mask = np.zeros_like(mask)
    left_lung_mask = np.zeros_like(mask)
    right_lung_mask = np.zeros_like(mask)
    mask_areas = [background_mask, left_lung_mask, right_lung_mask]

    # extract instances from mask
    labeled_mask = skimage.measure.label(mask)
    mask_labels = np.unique(labeled_mask)[1:]  # to remove background

    for label in mask_labels:

        one_instance_mask = (labeled_mask == label)
        one_instance_sum = one_instance_mask.sum()

        for area_label in [0, 1, 2]:  # 0 - backgound, 1 - left lung, 2 - right
            intersection = one_instance_mask * lung_areas[area_label]
            if intersection.sum() / one_instance_sum > intersection_threshold:
                mask_areas[area_label] += intersection

    return mask_areas


class LungRegionFeaturesExtractor:

    def __init__(self, resize = 256):
        self.lungs_labeled_mask = Resize(resize, resize).apply_to_mask(
            np.array(Image.open("../../data/masks012/average_mask.png"))
        )
        self.thresholds = np.arange(0, 1, 0.1)


    def extract(self, sample_mask, expert_mask):

            expert_masks = split_mask_to_zones(expert_mask, self.lungs_labeled_mask, intersection_threshold=0.3)
            sample_masks = split_mask_to_zones(sample_mask, self.lungs_labeled_mask, intersection_threshold=0.3)

            ious_1 = [get_iou(em, sm) for em, sm in zip(expert_masks, sample_masks)]
            ious_2 = [np.mean(get_instance_iou(em, sm, thresholds=self.thresholds, verbose=False)) for em, sm in
                          zip(expert_masks, sample_masks)]

            return np.array([
                get_iou(expert_mask, sample_mask),
                np.mean(get_instance_iou(expert_mask, sample_mask, thresholds=self.thresholds, verbose=0)),
                ious_1[0],
                ious_1[1],
                ious_1[2],
                ious_2[0],
                ious_2[1],
                ious_2[2],
            ])


