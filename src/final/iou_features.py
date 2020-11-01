import json
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import skimage
import skimage.io
import skimage.measure


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "Dataset")


def read_all_images():
    origin = {}
    expert = {}
    sample_1 = {}
    sample_2 = {}
    sample_3 = {}

    for filename in os.listdir(os.path.join(DATASET_DIR, "Origin")):
        tmp = filename.split(".")[0]
        origin[tmp] = plt.imread(os.path.join(DATASET_DIR, "Origin", filename))
        if len(origin[tmp].shape) == 3:
            origin[tmp] = origin[tmp][:, :, 0]
        expert[tmp] = plt.imread(os.path.join(DATASET_DIR, "Expert", f"{tmp}_expert.png"))
        sample_1[tmp] = plt.imread(os.path.join(DATASET_DIR, "sample_1", f"{tmp}_s1.png"))
        sample_2[tmp] = plt.imread(os.path.join(DATASET_DIR, "sample_2", f"{tmp}_s2.png"))
        sample_3[tmp] = plt.imread(os.path.join(DATASET_DIR, "sample_3", f"{tmp}_s3.png"))

    return origin, expert, sample_1, sample_2, sample_3


def read_all_labels():
    f = open(os.path.join(DATASET_DIR, "OpenPart.csv"))
    scores = {}
    next(f)
    for line in f:
        key, values = line.split(",", 1)
        key = key.split(".")[0]
        scores[key] = values.strip().split(",")
    return scores


def split_mask_to_zones(mask, lungs_labeled_mask, intersection_threshold=0.3):

    # split lungs mask to separate masks, one area for each mask
    lung_labels = [0, 1, 2]
    assert np.unique(lungs_labeled_mask).tolist() == lung_labels
    lung_areas = [lungs_labeled_mask == label for label in lung_labels]

    # placeholder for result
    background_mask = np.zeros_like(mask)
    left_lung_mask = np.zeros_like(mask)
    right_lung_mask = np.zeros_like(mask)

    mask_areas = [
        background_mask, left_lung_mask, right_lung_mask
    ]

    # extract instances from mask
    labeled_mask = skimage.measure.label(mask)
    mask_labels = np.unique(labeled_mask)[1:]  # to remove background

    for label in mask_labels:

        one_instance_mask = (labeled_mask == label)
        one_instance_sum = one_instance_mask.sum()

        for area_label in range(3):  # 0 - background, 1 - left lung, 2 - right
            intersection = one_instance_mask * lung_areas[area_label]
            if intersection.sum() / one_instance_sum > intersection_threshold:
                mask_areas[area_label] += intersection

    return mask_areas


# Precision helper function
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
    for t in thresholds:
        tp, fp, fn = precision_at(t, iou)
        if tp + fp + fn == 0:
            p = 1.
        else:
            p = (1 + beta**2) * tp / ((1 + beta**2) * tp + fp + beta**2 * fn + 1e-10)
        prec.append(p)

    return prec


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


def is_score_one(expert_mask, sample_mask, lungs_labeled_mask):
    iou = get_iou(expert_mask, sample_mask)
    return iou < 0.01


def is_score_five(expert_mask, sample_mask, lungs_labeled_mask):
    n_exp = len(np.unique(skimage.measure.label(expert_mask))) - 1
    n_sam = len(np.unique(skimage.measure.label(sample_mask))) - 1

    if n_exp == 1 and n_sam == 1 and get_iou(expert_mask, sample_mask) > 0.3:
        return True

    if n_exp == 2 and n_sam == 2 and np.mean(get_instance_iou(expert_mask, sample_mask, thresholds=[0.5])) == 1:
        return True

    return False


def get_heuristic_score(expert_mask, sample_mask, lungs_labeled_mask):
    if is_score_one(expert_mask, sample_mask, lungs_labeled_mask):
        return 1.

    if is_score_five(expert_mask, sample_mask, lungs_labeled_mask):
        return 5.


def main():
    # Read dataset
    origin, expert, sample_1, sample_2, sample_3 = read_all_images()
    scores = read_all_labels()
    keys = list(sorted(origin.keys()))

    lungs_labeled_mask = skimage.io.imread(os.path.join(DATASET_DIR, "..", "masks012", "average_mask.png"))

    # Calc heuristic scores
    heuristic_scores = {}
    for key in keys:
        for sample_i in range(3):
            samples_dict = [sample_1, sample_2, sample_3][sample_i]
            expert_mask = expert[key]
            sample_mask = samples_dict[key]
            h_score = get_heuristic_score(expert_mask, sample_mask, lungs_labeled_mask)
            if h_score is not None:
                heuristic_scores[f"{key}_s{sample_i}"] = h_score

    with open(os.path.join(DATASET_DIR, "..", "heuristic_scores.json"), "w") as f:
        json.dump(heuristic_scores, f, indent=4)

    # Calc IoU features
    thresholds = np.arange(0, 1, 0.1)
    data = []
    with tqdm(keys) as keys_:
        for key in keys_:
            expert_mask = expert[key]
            expert_masks = split_mask_to_zones(expert_mask, lungs_labeled_mask, intersection_threshold=0.3)

            for i, sample_dict in enumerate([sample_1, sample_2, sample_3]):
                sample_mask = sample_dict[key]

                # estimate ious per area
                sample_masks = split_mask_to_zones(sample_mask, lungs_labeled_mask, intersection_threshold=0.3)

                ious_1 = [get_iou(em, sm) for em, sm in zip(expert_masks, sample_masks)]
                ious_2 = [np.mean(get_instance_iou(em, sm, thresholds=thresholds, verbose=False)) for em, sm in zip(expert_masks, sample_masks)]
                score = scores[key][i] if key in scores else ""

                sample = dict(
                    key=key,
                    sample_i=i,
                    overall_iou=get_iou(expert_mask, sample_mask),
                    overall_instance_iou=np.mean(get_instance_iou(expert_mask, sample_mask, thresholds=thresholds, verbose=0)),
                    back_iou=ious_1[0],
                    left_lung_iou=ious_1[1],
                    right_lung_iou=ious_1[2],
                    back_instance_iou=ious_2[0],
                    left_lung_inst_iou=ious_2[1],
                    right_lung_inst_iou=ious_2[2],
                    target=score,
                )
                data.append(sample)

    df = pd.DataFrame(data, columns=data[0].keys())
    df.head()
    df.to_csv(os.path.join(DATASET_DIR, "..", "iou_features_v01.csv"), index=False)


if __name__ == "__main__":
    main()
