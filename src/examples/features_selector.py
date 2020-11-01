import catboost
import torch
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from torch import nn, Tensor

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, \
    GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from data.expert_algo import ExpertAlgo, ExpertAlgoTest
from data.mask_component_features import ComponentFeaturesExtractor, intersect_metric, contour_min_distance_metric, \
    coordinates_metric
import numpy as np

from data.transform import mix_transform

def make_test_set(sample_num: int, dataset):

    extractor = ComponentFeaturesExtractor([
        intersect_metric,
        contour_min_distance_metric,
        # coordinates_metric,
    ], aggregators=[np.mean, np.sum, np.max])

    X = []
    NC = 9

    for i in range(dataset.__len__()):
        data_i = dataset[i]
        expert = data_i["Expert"][0].numpy()
        sample = data_i[f"sample_{sample_num}"][0].numpy()
        iou = data_i[f"iou_{sample_num}"]

        xi = extractor.extract(sample, expert)
        if xi is None:
            xi = -np.ones(NC)
        else:
            NC = xi.shape[0]

        xi_inv = extractor.extract(expert, sample)

        if xi_inv is None:
            xi_inv = -np.ones(NC)

        xi = np.concatenate((xi, iou, xi_inv))
        X.append(xi.reshape(1, -1))

    return np.concatenate(X)


def make_train_set(sample_num: int, dataset):

    extractor = ComponentFeaturesExtractor([
        intersect_metric,
        contour_min_distance_metric,
        # coordinates_metric,
    ], aggregators=[np.mean, np.sum, np.max])

    X = []
    y = []
    NC = 9

    for i in range(dataset.__len__()):
        data_i = dataset[i]
        expert = data_i["Expert"][0].numpy()
        sample = data_i[f"sample_{sample_num}"][0].numpy()
        iou = data_i[f"iou_{sample_num}"]

        xi = extractor.extract(sample, expert)
        if xi is None:
            xi = -np.ones(NC)
        else:
            NC = xi.shape[0]

        xi_inv = extractor.extract(expert, sample)

        if xi_inv is None:
            xi_inv = -np.ones(NC)

        yi = int(data_i[f"label_{sample_num}"].item())

        xi = np.concatenate((xi, iou, xi_inv))
        X.append(xi.reshape(1, -1))
        y.append(yi)

    return np.concatenate(X), np.array(y)


def make_complete_train_set(dataset, permutation: np.ndarray = None):

    X1,y1 = make_train_set(sample_num=1, dataset=dataset)
    X2,y2 = make_train_set(sample_num=2, dataset=dataset)
    X3,y3 = make_train_set(sample_num=3, dataset=dataset)

    # X1u = np.concatenate((X1, y2.reshape(-1, 1), y3.reshape(-1, 1)), axis=1)
    # X2u = np.concatenate((X2, y1.reshape(-1, 1), y3.reshape(-1, 1)), axis=1)
    # X3u = np.concatenate((X3, y1.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)

    X = np.concatenate((X1, X2, X3))
    y = np.concatenate((y1, y2, y3))

    if permutation is not None:
        X = X[permutation]
        y = y[permutation]

    return X, y

def select_features(X, y):

    clf = ExtraTreesClassifier(n_estimators=100)
    clf = clf.fit(X, y)

    print(clf.feature_importances_)
    model = SelectFromModel(clf, prefit=True, threshold=0.04)
    X_new = model.transform(X)
    print(X_new.shape)

    return model


if __name__ == "__main__":

    dataset_train = ExpertAlgo(transforms=mix_transform(256))
    dataset_test = ExpertAlgoTest(transforms=mix_transform(256))

    X1, y1 = make_complete_train_set(dataset_train)
    selection_model = select_features(X1, y1)
    N = X1.shape[0]
    N_train = int(0.9 * N)

    scores = []

    NFOLDS = 20
    for _ in range(NFOLDS):

        # clf = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1,
        #                                  max_depth=2, random_state=0, loss='ls', criterion="mae")
        # clf = RandomForestRegressor(n_estimators=50)
        clf = catboost.CatBoostRegressor(
            iterations=500,
            silent=True,
            learning_rate=0.05,
            depth=3,
            loss_function='MAE'
        )

        permutation = np.random.permutation(N)

        X_fold, y_fold = [], []
        X_test, y_test = None, None

        for k in range(1):
            # Xk, yk = make_complete_train_set(dataset_train, permutation)
            Xk, yk = X1[permutation], y1[permutation]
            Xk = selection_model.transform(Xk)
            X_fold.append(Xk[0:N_train])
            y_fold.append(yk[0:N_train])
            if k == 0:
                X_test, y_test = Xk[N_train:], yk[N_train:]

        X_fold, y_fold = np.concatenate(X_fold), np.concatenate(y_fold)

        clf.fit(X_fold, y_fold * 1.0)

        y_pred: np.ndarray = np.round(clf.predict(X_test))

        mae = np.mean(np.abs(y_test - y_pred))
        print(mae)
        print(y_pred)
        scores.append(mae)

    scores = np.array(scores)
    print("score", np.mean(scores), np.std(scores))

