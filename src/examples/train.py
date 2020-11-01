import catboost
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold, cross_val_predict
from data.expert_algo import ExpertAlgo, ExpertAlgoTest
from features_extractor.mask_component_features import ComponentFeaturesExtractor, intersect_metric, \
    contour_min_distance_metric, intersect_nearest
import numpy as np

from data.transform import simple_transform
from features_extractor.lung_region_features import LungRegionFeaturesExtractor
from tuner import GoldTuner


def make_test_set(sample_num: int, dataset):

    extractor = ComponentFeaturesExtractor([
        intersect_metric,
        contour_min_distance_metric,
        # coordinates_metric,
    ], aggregators=[np.mean, np.sum, np.max])

    extractor_2 = LungRegionFeaturesExtractor(resize=256)

    X = []
    NC = 9

    for i in range(dataset.__len__()):
        data_i = dataset[i]
        expert = data_i["Expert"][0].numpy()
        sample = data_i[f"sample_{sample_num}"][0].numpy()
        # iou = data_i[f"iou_{sample_num}"]
        iou = extractor_2.extract(sample, expert)

        # iou2 = extractor_2.extract(sample, expert)
        # print(iou - iou2)

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


def make_complete_test_set(dataset):

    X1 = make_test_set(sample_num=1, dataset=dataset_test)
    X2 = make_test_set(sample_num=2, dataset=dataset_test)
    X3 = make_test_set(sample_num=3, dataset=dataset_test)

    return np.concatenate((X1, X2, X3))


def make_train_set(sample_num: int, dataset):

    extractor = ComponentFeaturesExtractor([
        intersect_metric,
        contour_min_distance_metric,
        # intersect_nearest,
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
    model = SelectFromModel(clf, prefit=True, threshold=0.035)
    X_new = model.transform(X)
    print(X_new.shape)

    return model


def cross_val(Xt, yt):

    reg = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1,
                                     max_depth=2, random_state=0, loss='ls', criterion="mae")
    # reg = RandomForestRegressor(n_estimators=50)
    # reg = catboost.CatBoostRegressor(
    #     iterations=500,
    #     silent=True,
    #     learning_rate=0.05,
    #     depth=3,
    #     loss_function='MAE'
    # )

    cv = KFold(n_splits=10).split(Xt, yt)

    cv_pred = cross_val_predict(
        reg,
        Xt, yt,
        cv=cv
    )

    return mean_absolute_error(yt, np.round(cv_pred))


if __name__ == "__main__":

    dataset_train = ExpertAlgo(transforms=simple_transform(256))
    dataset_test = ExpertAlgoTest(transforms=simple_transform(256))

    Xt, yt = make_complete_train_set(dataset_train)
    selection_model = select_features(Xt, yt)
    Xt = selection_model.transform(Xt)

    # reg = catboost.CatBoostRegressor(
    #     iterations=500,
    #     silent=True,
    #     learning_rate=0.05,
    #     depth=3,
    #     loss_function='MAE'
    # )
    reg = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1,
                                    max_depth=2, random_state=0, loss='ls', criterion="mae")

    reg.fit(Xt, yt)

    score = cross_val(Xt, yt)
    print("score", score)

    X_test = selection_model.transform(make_complete_test_set(dataset_test))
    perm = np.random.permutation(X_test.shape[0])
    X_test = X_test[perm][:50]

    y_test: np.ndarray = reg.predict(X_test)

    tuner = GoldTuner(y_test.tolist(), device="cpu", rule_eps=0.01, radius=1.2)

    for _ in range(1000):

        Xtt = np.concatenate((Xt, X_test))
        ytt = np.concatenate((yt, tuner.get_coef().numpy()))

        score = cross_val(Xtt, ytt)
        print("score", score)
        tuner.update(score)

