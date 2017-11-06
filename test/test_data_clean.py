"""Simple tests for data cleaning."""
import numpy as np

from atoml.utilities.clean_data import remove_outliers


def outlier_test():
    f = np.arange(200).reshape(50, 4)

    t = np.random.random_sample((50,))
    t[2] += 200000.
    t[0] -= 200000.

    d = remove_outliers(features=f, targets=t, constraint=None)

    assert np.shape(d['features']) != np.shape(f)
    assert np.shape(d['targets']) != np.shape(t)
    assert np.shape(d['features'])[0] == np.shape(d['targets'])[0]
