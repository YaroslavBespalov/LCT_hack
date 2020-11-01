import os

import numpy as np

import catboost

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict

from expert_dataset import ExpertDataset
from mask_component_features import ComponentFeaturesExtractor, intersect_metric, contour_min_distance_metric


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def make_train_set(dataset, sample_num: int):
    extractor = ComponentFeaturesExtractor([
        intersect_metric,
        contour_min_distance_metric
    ], aggregators=[np.mean, np.sum, np.max])

    X = []
    y = []
    NC = 9

    for i in range(dataset.__len__()):
        data_i = dataset[i]
        expert = data_i['Expert'][0].numpy()
        sample = data_i[f'sample_{sample_num}'][0].numpy()
        iou = data_i[f'iou_{sample_num}']

        xi = extractor.extract(sample, expert)
        if xi is None:
            xi = -np.ones(NC)
        else:
            NC = xi.shape[0]

        xi_inv = extractor.extract(expert, sample)

        if xi_inv is None:
            xi_inv = -np.ones(NC)

        yi = int(data_i[f'label_{sample_num}'].item())

        xi = np.concatenate((xi, iou, xi_inv))
        X.append(xi.reshape(1, -1))
        y.append(yi)

    return np.concatenate(X), np.array(y)


def make_complete_train_set(dataset, permutation: np.ndarray = None):
    X1, y1 = make_train_set(dataset, sample_num=1)
    X2, y2 = make_train_set(dataset, sample_num=2)
    X3, y3 = make_train_set(dataset, sample_num=3)

    X = np.concatenate((X1, X2, X3))
    y = np.concatenate((y1, y2, y3))

    if permutation is not None:
        X = X[permutation]
        y = y[permutation]

    return X, y


def select_features(X, y):
    clf = ExtraTreesClassifier(n_estimators=1000)
    clf = clf.fit(X, y)

    print('Feature count:', X.shape[1])
    print('Feature importances:')
    print(clf.feature_importances_)

    model = SelectFromModel(clf, prefit=True, threshold=0.04)
    X_new = model.transform(X)
    print('New feature count:', X_new.shape[1])
    print()

    return X_new


def main():
    dataset = ExpertDataset(path=os.path.join(SCRIPT_DIR, '..', '..', 'data'), enable_transforms=False)

    X, y = make_complete_train_set(dataset)
    X_selected = select_features(X, y)

    reg = catboost.CatBoostRegressor(
        iterations=500,
        silent=True,
        learning_rate=0.05,
        depth=3,
        loss_function='MAE'
    )

    cv = KFold(n_splits=10).split(X_selected, y)

    cv_pred = cross_val_predict(reg, X_selected, y, cv=cv)

    print('Cross validation predictions:')
    print(cv_pred)
    print('MAE:', mean_absolute_error(y, np.round(cv_pred)))


if __name__ == '__main__':
    main()
