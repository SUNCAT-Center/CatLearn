"""Simple tests for data cleaning."""
import numpy as np

from atoml.utilities import clean_data as clean


def outlier_test():
    """Test outlier removal from toy features."""
    f = np.arange(200).reshape(50, 4)

    t = np.random.random_sample((50,))
    t[2] += 200000.
    t[0] -= 200000.

    d = clean.remove_outliers(features=f, targets=t, constraint=None)

    assert np.shape(d['features']) != np.shape(f)
    assert np.shape(d['targets']) != np.shape(t)
    assert np.shape(d['features'])[0] == np.shape(d['targets'])[0]


def variance_test():
    """Test cleaning zero variace features."""
    features = np.random.random_sample((50, 5))
    features[:, 1:2] = 109.982
    features = clean.clean_variance(features)['train']

    assert np.shape(features) == (50, 4)


def inf_test():
    """Test cleaning inf variable features."""
    features = np.random.random_sample((50, 5))
    features[1][0] = np.inf
    features = clean.clean_infinite(features)['train']

    assert np.shape(features) == (50, 4)


if __name__ == '__main__':
    outlier_test()
    variance_test()
    inf_test()
