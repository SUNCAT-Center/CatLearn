"""Functions for feature engineering."""
from __future__ import absolute_import
from __future__ import division

import numpy as np
from itertools import combinations_with_replacement


def single_transform(A):
    """Perform single variable transform x^2, x^0.5 and log(x).

    Parameters
    ----------
    A : array
        n x m matrix, where n is the number of training examples and m is the
        number of features.

    Returns
    -------
    new_features : array
        The n x m*3 matrix of new features.
    """
    scale = A + np.abs(np.min(A, axis=0)) + 1.
    new_features = np.concatenate((np.square(A), np.sqrt(scale)), axis=1)
    new_features = np.concatenate((new_features, np.log(scale)), axis=1)

    return new_features


def get_order_2(A):
    """Get all combinations x_ij = x_i * x_j, where x_i,j are features.

    The sorting order in dimension 0 is preserved.

    Parameters
    ----------
    A : array
        n x m matrix, where n is the number of training examples and m is the
        number of features.

    Returns
    -------
    new_features : array
        The n x triangular(m) matrix of new features.
    """
    shapeA = np.shape(A)
    nfi = 0
    new_features = np.zeros([shapeA[0], sum(range(shapeA[1] + 1))])
    for f1 in range(shapeA[1]):
        for f2 in range(f1, shapeA[1]):
            new_feature = A[:, f1] * A[:, f2]
            new_features[:, nfi] = new_feature
            nfi += 1
    return new_features


def get_div_order_2(A):
    """Get all combinations x_ij = x_i / x_j, where x_i,j are features.

    The sorting order in dimension 0 is preserved. If a value is 0, Inf is
    returned.

    Parameters
    ----------
    A : array
        n x m matrix, where n is the number of training examples and m is the
        number of features.

    Returns
    -------
    new_features : array
        The n x m**2 matrix of new features.
    """
    shapeA = np.shape(A)
    nfi = 0
    # Preallocate:
    new_features = np.zeros([shapeA[0], shapeA[1]**2])
    for f1 in range(shapeA[1]):
        for f2 in range(shapeA[1]):
            if f1 != f2:
                new_feature = np.true_divide(A[:, f1], A[:, f2])
                new_features[:, nfi] = new_feature
                nfi += 1
    return new_features


def get_labels_order_2(l, div=False):
    """Get all combinations ij, where i,j are feature labels.

    Parameters
    ----------
    x : list
        Length m vector, where m is the number of features.

    Returns
    -------
    new_features : list
        List of new feature names.
    """
    L = len(l)
    new_features = []
    if div:
        op = '_div_'
        s = 0
    else:
        op = '_x_'
    for f1 in range(L):
        if not div:
            s = f1
        for f2 in range(s, L):
            new_features.append(l[f1] + op + l[f2])
    return new_features


def get_order_2ab(A, a, b):
    """Get all combinations x_ij = x_i**a * x_j**b, where x_i,j are features.

    The sorting order in dimension 0 is preserved.

    Parameters
    ----------
    A : array
        n x m matrix, where n is the number of training examples and m is the
        number of features.
    a : float

    b : float

    Returns
    -------
    new_features : array
        The n x triangular(m) matrix of new features.
    """
    shapeA = np.shape(A)
    nfi = 0
    new_features = np.zeros([shapeA[0], sum(range(shapeA[1] + 1))])
    for f1 in range(shapeA[1]):
        for f2 in range(f1, shapeA[1]):
            new_feature = A[:, f1]**a * A[:, f2]**b
            new_features[:, nfi] = new_feature
            nfi += 1
    return new_features


def get_labels_order_2ab(l, a, b):
    """Get all combinations ij, where i,j are feature labels.

    Parameters
    ----------
    x : list
        Length m vector, where m is the number of features.

    Returns
    -------
    new_features : list
        List of new feature names.
    """
    L = len(l)
    new_features = []
    for f1 in range(L):
        for f2 in range(f1, L):
            new_features.append(l[f1] + '_' + str(a) + '_x_' + l[f2] + '_' +
                                str(b))
    return new_features


def get_ablog(A, a, b):
    """Get all combinations x_ij = a*log(x_i) + b*log(x_j).

    The sorting order in dimension 0 is preserved.

    Parameters
    ----------
    A : array
        An n x m matrix, where n is the number of training examples and m is
        the number of features.
    a : float

    b : float

    Returns
    -------
    new_features : array
        The n x triangular(m) matrix of new features.
    """
    shapeA = np.shape(A)
    shift = np.abs(np.min(A, axis=0)) + 1.
    A += shift
    nfi = 0
    new_features = np.zeros([shapeA[0], sum(range(shapeA[1] + 1))])
    for f1 in range(shapeA[1]):
        for f2 in range(f1, shapeA[1]):
            new_feature = a * np.log(A[:, f1]) + b * np.log(A[:, f2])
            new_features[:, nfi] = new_feature
            nfi += 1
    return new_features


def get_labels_ablog(l, a, b):
    """Get all combinations ij, where i,j are feature labels.

    Parameters
    ----------
    a : float
    b : float

    Returns
    -------
    new_features : list
        List of new feature names.
    """
    L = len(l)
    new_features = []
    for f1 in range(L):
        for f2 in range(f1, L):
            # TODO Better string formatting with numbers.
            new_features.append('log' + str(a) + '_' + l[f1] + 'log' + str(b) +
                                '_' + l[f2])
    return new_features


def _separate_list(p):
    """Routine to split any list.

    List is split into all possible combinations of two lists which, combined,
    contain all elements.

    Parameters
    ----------
    p : list
        The list to be split.

    Returns
    -------
    combinations : list
        A list containing num_combinations elements, each of which is a tuple.
        Each tuple contains two elements, each of which is a list. These two
        tuple elements have no intersection, their union is p.
    """
    num_elements = len(p)
    num_combinations = (2**num_elements - 2) / 2
    key = '0%db' % num_elements
    combinations = []
    for i in range(1, num_combinations + 1):
        bin_str = format(i, key)
        left = []
        right = []
        for j in range(num_elements):
            if bin_str[j] == '0':
                left.append(p[j])
            elif bin_str[j] == '1':
                right.append(p[j])
        combinations.append((left, right))
    return combinations


def _decode_key(p, key):
    """Routine to decode a "key" as implemented in generate_features.

    These keys are used to avoid duplicate terms in numerator and denominator.

    Parameters
    ----------
    p : list
        The list of input features provided by the user.
    key : string
        A string containing a composite term, where each original feature in p
        is represented by its index.

        Example:
        The term given by p[0]*p[1]*p[1]*p[4] would have the key "0*1*1*4"

    Returns
    -------
    p_prime : string
        A string containing the composite term as a function of the original
        input features.
    """
    p = [str(i) for i in p]
    elements = key.split('*')
    translated_elements = [p[int(i)] for i in elements]
    unique_elements = list(set(translated_elements))
    unique_elements.sort()
    count_elements = {}
    for ele in unique_elements:
        count_elements[ele] = 0
    for ele in translated_elements:
        count_elements[ele] += 1
    ele_list = []
    for ele in unique_elements:
        count = count_elements[ele]
        if count == 1:
            ele_list.append(ele)
        if count >= 2:
            ele_list.append(ele + '^%d' % count)
    p_prime = '*'.join(ele_list)
    return p_prime


def generate_positive_features(p, N, exclude=False, s=False):
    """Generate list of polynomial combinations in list p up to order N.

    Example:
    p = (a,b,c) ; N = 3

    returns (order not preserved)
    [a*a*a, a*a*b, a*a*c, a*b*b, a*b*c, a*c*c, b*b*b, b*b*c, b*c*c, c*c*c, a*a,
    a*b, a*c, b*b, b*c, c*c, a, b, c]

    Parameters
    ----------
    p : list
        Features to be combined.
    N : integer
        The maximum polynomial coefficient for combinations. Must be
        non-negative.
    exclude : bool
        Set True to avoid returning 1 to represent the zeroth power. Default is
        False.
    s : bool
        Set True to return a list of strings and False to evaluate each element
        in the list. Default is False.

    Returns
    -------
    all_powers : list
        A list of combinations of the input features to meet the required
        specifications.
    """
    if N == 0 and s:
        return ['1']
    elif N == 0 and not s:
        return [1]
    elif N == 1 and not exclude and s:
        return p + ["1"]
    elif N == 1 and not exclude and not s:
        return p + [1]
    if N == 1 and exclude:
        return p
    else:
        all_powers = []
        p = [str(i) for i in p]
        for i in range(N, 0, -1):
            thisPower = combinations_with_replacement(p, i)
            tuples = list(thisPower)
            ntuples = len(tuples)
            thisPowerFeatures = ['*'.join(tuples[j]) for j in range(ntuples)]
            if not s:
                thisPowerFeatures = [eval(j) for j in thisPowerFeatures]
            all_powers.append(thisPowerFeatures)
        if not exclude:
            if s:
                all_powers.append(['1'])
            else:
                all_powers.append([1])
        all_powers = [item for sublist in all_powers for item in sublist]
        return all_powers


def generate_features(p, max_num=2, max_den=1, log=False, sqrt=False,
                      exclude=False, s=False):
    """Generate composite features from a combination of input features.

    developer note: This is currently scales *quite slowly* with max_den.
    There's surely a better way to do this, but it's apparently currently
    functional.

    Parameters
    ----------
    p : list
        User-provided list of physical features to be combined.
    max_num : integer
        The maximum order of the polynomial in the numerator of the composite
        features. Must be non-negative.
    max_den : integer
        The maximum order of the polynomial in the denominator of the composite
        features. Must be non-negative.
    log : boolean (not currently supported)
        Set to True to include terms involving the logarithm of the input
        features. Default is False.
    sqrt : boolean (not currently supported)
        Set to True to include terms involving the square root of the input
        features. Default is False.
    exclude : bool
        Set exclude=True to avoid returning 1 to represent the zeroth power.
        Default is False.
    s: bool
        Set True to return a list of strings and False to evaluate each element
        in the list. Default is False.

    Returns
    -------
    features : list
        A list of combinations of the input features to meet the required
        specifications.
    """
    if max_den == 0:
        return generate_positive_features(p, max_num, exclude=exclude, s=s)
    if max_num == 0:
        dup_feature_keys = generate_positive_features(p, max_den,
                                                      exclude=exclude, s=True)
        features = []
        for key in dup_feature_keys:
            val = '1/(' + key + ')'
            features.append(val)
        if not s:
            features = [eval('1.*' + i) for i in features]
        return features
    else:
        num_p = len(p)
        p_str = [str(i) for i in range(num_p)]
        features = []
        feature_keys = generate_positive_features(p_str, max_num, exclude=True,
                                                  s=True)
        dup_feature_keys = generate_positive_features(p_str, max_den,
                                                      exclude=True, s=True)
        for key1 in feature_keys:
            l1 = key1.split('*')
            for key2 in dup_feature_keys:
                l2 = key2.split('*')
                intersect = list(set.intersection(set(l1), set(l2)))
                if not intersect:
                    val = _decode_key(p, key1) + '/(' + \
                        _decode_key(p, key2) + ')'
                    features.append(val)
        for key1 in feature_keys:
            features.append(_decode_key(p, key1) + '/(1)')
        for key2 in dup_feature_keys:
            features.append('1/(' + _decode_key(p, key2) + ')')
        if not exclude:
            features.append('1')
        if not s:
            features = [eval('1.*' +
                             str.replace(i, '^', '**')) for i in features]
        return features
