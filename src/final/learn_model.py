import json
import os

import numpy as np

import catboost

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict

from expert_dataset import ExpertDataset, ExpertDatasetTest
from mask_component_features import ComponentFeaturesExtractor, intersect_metric, contour_min_distance_metric


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'data')


def make_set(dataset, sample_num: int):
    extractor = ComponentFeaturesExtractor([
        intersect_metric,
        contour_min_distance_metric
    ], aggregators=[np.mean, np.sum, np.max])

    keys = []
    X = []
    y = []
    NC = 9

    for i in range(dataset.__len__()):
        data_i = dataset[i]
        expert = data_i['Expert'][0].numpy().copy()
        sample = data_i[f'sample_{sample_num}'][0].numpy().copy()
        iou = data_i[f'iou_{sample_num}']

        xi = extractor.extract(sample, expert)
        if xi is None:
            xi = -np.ones(NC)
        else:
            NC = xi.shape[0]

        xi_inv = extractor.extract(expert, sample)

        if xi_inv is None:
            xi_inv = -np.ones(NC)

        try:
            yi = int(data_i[f'label_{sample_num}'].item())
        except KeyError:
            yi = None

        xi = np.concatenate((xi, iou, xi_inv))
        keys.append(f'{data_i["key"]}_s{sample_num - 1}')
        X.append(xi.reshape(1, -1))
        y.append(yi)

    return keys, np.concatenate(X), np.array(y)


def make_complete_train_set(dataset):
    keys1, X1, y1 = make_set(dataset, sample_num=1)
    keys2, X2, y2 = make_set(dataset, sample_num=2)
    keys3, X3, y3 = make_set(dataset, sample_num=3)

    keys = keys1 + keys2 + keys3
    X = np.concatenate((X1, X2, X3))
    y = np.concatenate((y1, y2, y3))

    return keys, X, y


def make_complete_test_set(dataset):
    keys1, X1, _ = make_set(dataset, sample_num=1)
    keys2, X2, _ = make_set(dataset, sample_num=2)
    keys3, X3, _ = make_set(dataset, sample_num=3)

    return keys1 + keys2 + keys3, np.concatenate((X1, X2, X3))


def select_features(X, y):
    clf = ExtraTreesClassifier(n_estimators=1000, random_state=117)
    clf = clf.fit(X, y)

    print('Feature count:', X.shape[1])
    print('Feature importances:')
    print(clf.feature_importances_)

    model = SelectFromModel(clf, prefit=True, threshold=0.04)
    X_new = model.transform(X)
    print('New feature count:', X_new.shape[1])
    print()

    return model


def postprocess(keys, pred):
    with open(os.path.join(DATA_DIR, 'heuristic_scores.json'), 'r') as f:
        heuristic_scores = json.load(f)

    result = np.zeros_like(pred)
    for i, key in enumerate(keys):
        result[i] = np.round(heuristic_scores.get(key, pred[i]))

    return result


def main():
    # Read dataset
    dataset_train = ExpertDataset(path=DATA_DIR)
    dataset_test = ExpertDatasetTest(path=DATA_DIR)

    # Calc features
    keys_train, X_train, y_train = make_complete_train_set(dataset_train)
    keys_test, X_test = make_complete_test_set(dataset_test)

    # Feature selection
    select_model = select_features(X_train, y_train)
    X_train_selected = select_model.transform(X_train)
    X_test_selected = select_model.transform(X_test)

    # Cross validation
    reg = catboost.CatBoostRegressor(
        iterations=500,
        silent=True,
        learning_rate=0.05,
        depth=3,
        loss_function='MAE'
    )

    cv = KFold(n_splits=10).split(X_train_selected, y_train)
    cv_pred = cross_val_predict(reg, X_train_selected, y_train, cv=cv)
    cv_pred_postprocessed = postprocess(keys_train, cv_pred)

    print('Cross validation predictions:')
    print(cv_pred_postprocessed)
    print('Cross validation MAE:', mean_absolute_error(y_train, cv_pred_postprocessed))
    print()

    # Full train
    reg.fit(X_train_selected, y_train)

    # Predict
    test_pred = reg.predict(X_test_selected)
    test_pred_postprocessed = postprocess(keys_test, test_pred)
    print(test_pred_postprocessed)


if __name__ == '__main__':
    main()
