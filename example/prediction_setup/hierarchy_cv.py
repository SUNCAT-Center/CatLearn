"""A simple example for the hierarchy CV function."""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import AffinityPropagation

from atoml.cross_validation import HierarchyValidation
from atoml.ridge_regression import RidgeRegression
from atoml.feature_preprocess import standardize
from atoml.predict import target_standardize

# Define the hierarchy cv class method.
hv = HierarchyValidation(db_name='../../data/train_db.sqlite',
                         table='FingerVector',
                         file_name='hierarchy')
# Split the data into subsets.
hv.split_index(min_split=1000, max_split=8000)
# Load data back in from save file.
ind = hv.load_split()


def predict(train_features, train_targets, test_features, test_targets,
            features):
    """Function to perform the prediction."""
    data = {}

    # Set how many features to include in the model.
    train_features = train_features[:, :features]
    test_features = test_features[:, :features]

    store_train = train_features
    store_test = test_features
    store_target = train_targets

    af = AffinityPropagation().fit(train_features)
    tc = af.predict(test_features)
    dist = []
    abds = []
    for i in test_features:
        md = float('inf')
        s = None
        for j in af.cluster_centers_:
            d = np.linalg.norm(j - i) / 370
            if d < md:
                md = d
                s = (np.linalg.norm(j) - np.linalg.norm(i)) / 370
        abds.append(s)
        dist.append(d)

    fig, ax = plt.subplots(nrows=2, ncols=3)

    ax[1][0].scatter(abds, test_targets, alpha=0.5, c=tc,
                     cmap=plt.cm.nipy_spectral)
    ax[1][0].set_xlabel('distance measure')
    ax[1][0].set_ylabel('target')

    # Set up the ridge regression function.
    rr = RidgeRegression(W2=None, Vh=None, cv='loocv')
    b = rr.find_optimal_regularization(X=train_features, Y=train_targets)
    coef = rr.RR(X=train_features, Y=train_targets, omega2=b)[0]

    # Test the model.
    sumd = 0.
    err = []
    for tf, tt in zip(test_features, test_targets):
        p = np.dot(coef, tf)
        sumd += (p - tt) ** 2
        e = ((p - tt) ** 2) ** 0.5
        err.append(e)
    error = (sumd / len(test_features)) ** 0.5

    ax[0][0].scatter(dist, err, alpha=0.5, c=tc, cmap=plt.cm.nipy_spectral)
    ax[0][0].set_title('original: ' + str(round(error, 3)))
    ax[0][0].set_xlabel('distance measure')
    ax[0][0].set_ylabel('prediction error')

    std = standardize(train_matrix=train_features, test_matrix=test_features)
    train_features = std['train']
    test_features = std['test']

    af = AffinityPropagation().fit(train_features)
    tc = af.predict(test_features)
    dist = []
    abds = []
    for i in test_features:
        md = float('inf')
        s = None
        for j in af.cluster_centers_:
            d = np.linalg.norm(j - i) / 370
            if d < md:
                md = d
                s = (np.linalg.norm(j) - np.linalg.norm(i)) / 370
        abds.append(s)
        dist.append(d)

    ax[1][1].scatter(abds, test_targets, alpha=0.5, c=tc,
                     cmap=plt.cm.nipy_spectral)
    ax[1][1].set_xlabel('distance measure')
    ax[1][1].set_ylabel('target')

    ts = target_standardize(train_targets)
    train_targets = ts['target']

    # Set up the ridge regression function.
    rr = RidgeRegression(W2=None, Vh=None, cv='loocv')
    b = rr.find_optimal_regularization(X=train_features, Y=train_targets)
    coef = rr.RR(X=train_features, Y=train_targets, omega2=b)[0]

    # Test the model.
    sumd = 0.
    err = []
    for tf, tt in zip(test_features, test_targets):
        p = (np.dot(coef, tf) * ts['std']) + ts['mean']
        sumd += (p - tt) ** 2
        e = ((p - tt) ** 2) ** 0.5
        err.append(e)
    error = (sumd / len(test_features)) ** 0.5

    ax[0][1].scatter(dist, err, alpha=0.5, c=tc, cmap=plt.cm.nipy_spectral)
    ax[0][1].set_title('original: ' + str(round(error, 3)))
    ax[0][1].set_xlabel('distance measure')
    ax[0][1].set_ylabel('prediction error')

    globaldata = np.concatenate((store_train, store_test), axis=0)
    m = np.mean(globaldata, axis=0)
    s = np.std(globaldata, axis=0)
    np.place(s, s == 0., [1.])

    train_features = (np.asarray(store_train) - m) / s
    test_features = (np.asarray(store_test) - m) / s

    af = AffinityPropagation().fit(train_features)
    tc = af.predict(test_features)
    dist = []
    abds = []
    for i in test_features:
        md = float('inf')
        s = None
        for j in af.cluster_centers_:
            d = np.linalg.norm(j - i) / 370
            if d < md:
                md = d
                s = (np.linalg.norm(j) - np.linalg.norm(i)) / 370
        abds.append(s)
        dist.append(d)

    ax[1][2].scatter(abds, test_targets, alpha=0.5, c=tc,
                     cmap=plt.cm.nipy_spectral)
    ax[1][2].set_xlabel('distance measure')
    ax[1][2].set_ylabel('target')

    ts = target_standardize(store_target)
    train_targets = ts['target']

    # Set up the ridge regression function.
    rr = RidgeRegression(W2=None, Vh=None, cv='loocv')
    b = rr.find_optimal_regularization(X=train_features, Y=train_targets)
    coef = rr.RR(X=train_features, Y=train_targets, omega2=b)[0]

    # Test the model.
    sumd = 0.
    err = []
    for tf, tt in zip(test_features, test_targets):
        p = (np.dot(coef, tf) * ts['std']) + ts['mean']
        sumd += (p - tt) ** 2
        e = ((p - tt) ** 2) ** 0.5
        err.append(e)
    error = (sumd / len(test_features)) ** 0.5

    ax[0][2].scatter(dist, err, alpha=0.5, c=tc, cmap=plt.cm.nipy_spectral)
    ax[0][2].set_title('original: ' + str(round(error, 3)))
    ax[0][2].set_xlabel('distance measure')
    ax[0][2].set_ylabel('prediction error')

    plt.show()

    exit()

    return data


# Make the predictions for each subset.
res = hv.split_predict(index_split=ind, predict=predict, features=370)
# Print out the errors.
for i in res:
    print(i[0], i[1])
