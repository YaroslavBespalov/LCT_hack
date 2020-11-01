import torch
from sklearn.ensemble import GradientBoostingRegressor
from torch import nn, Tensor

from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import numpy as np
from data.expert_algo import ExpertAlgoTest, ExpertAlgo
from data.transform import mix_transform, simple_transform
from examples.features_selector import make_complete_train_set, select_features, make_test_set


def colorize(img, color):
    color = np.array(color)
    return np.stack(3 * [img], axis=-1) * color

def blend(a, b ,c):
    return np.clip(a + b * 0.5 + c * 0.5, 0, 1)


def viz_test(dataset, targets):

    for i in range(dataset.__len__()):

        data_i = dataset[i]
        image = data_i["image"][0].numpy()
        expert = data_i["Expert"][0].numpy()
        sample_1 = data_i[f"sample_1"][0].numpy()
        sample_2 = data_i[f"sample_2"][0].numpy()
        sample_3 = data_i[f"sample_3"][0].numpy()

        plt.figure(figsize=(20, 10))
        plt.subplot(131)
        plt.title(f"{targets[i]['sample_1']}")
        plt.imshow(
            blend(
                colorize(image, [1, 1, 1]),
                colorize(expert, [0, 1, 0]),
                colorize(sample_1, [1, 0, 0])
            )
        )

        plt.subplot(132)
        plt.title(f"{targets[i]['sample_2']}")
        plt.imshow(
            blend(
                colorize(image, [1, 1, 1]),
                colorize(expert, [0, 1, 0]),
                colorize(sample_2, [1, 0, 0])
            )
        )

        plt.subplot(133)
        plt.title(f"{targets[i]['sample_3']}")
        plt.imshow(
            blend(
                colorize(image, [1, 1, 1]),
                colorize(expert, [0, 1, 0]),
                colorize(sample_3, [1, 0, 0])
            )
        )

        # plt.show()
        plt.savefig(f"../../../test_plot/{i}.png")
        plt.close()


if __name__ == "__main__":

    dataset_train = ExpertAlgo(transforms=mix_transform(256))

    X_fold, y_fold = [], []

    for k in range(3):
        Xk, yk = make_complete_train_set(dataset_train)
        X_fold.append(Xk)
        y_fold.append(yk)

    X_fold, y_fold = np.concatenate(X_fold), np.concatenate(y_fold)

    selection_model = select_features(X_fold, y_fold)

    clf = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1,
                                    max_depth=2, random_state=0, loss='ls', criterion="mae")

    dataset_test = ExpertAlgoTest(transforms=simple_transform(256))

    X1 = make_test_set(sample_num=1, dataset=dataset_test)
    X2 = make_test_set(sample_num=2, dataset=dataset_test)
    X3 = make_test_set(sample_num=3, dataset=dataset_test)

    clf.fit(X_fold, y_fold * 1.0)

    y_pred_1: np.ndarray = clf.predict(X1)
    y_pred_2: np.ndarray = clf.predict(X2)
    y_pred_3: np.ndarray = clf.predict(X3)

    targets = []

    for i in range(dataset_test.__len__()):

        targets.append({
            "sample_1": np.round(y_pred_1[i]),
            "sample_2": np.round(y_pred_2[i]),
            "sample_3": np.round(y_pred_3[i])
        })

    viz_test(dataset_test, targets)
