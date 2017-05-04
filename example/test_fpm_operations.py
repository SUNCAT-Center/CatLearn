""" Script to test feature expansion """
from __future__ import print_function

import numpy as np

from atoml import fpm_operations as fpm


data = {'Ag': {'a': 3., 'b': 5., 'c': 1.},
        'Pt': {'a': 2., 'b': 1., 'c': 2.},
        'Rh': {'a': 1., 'b': 3., 'c': 4.}}

examples = list(data.keys())
num_examples = len(examples)
features = data[examples[0]].keys()

max_num = 3
max_den = 1
s = True
exclude = False

composite_features = fpm.generate_features(features, max_num=max_num,
                                           max_den=max_den, exclude=exclude,
                                           s=s)
num_composite_features = len(composite_features)

composite_fp_matrix = []
for i in range(num_examples):
    example = examples[i]
    example_data = []
    for j in range(num_composite_features):
        eval_string = composite_features[j]
        eval_string = eval_string.replace('^', '**')
        for feature in features:
            eval_string = eval_string.replace(feature,
                                              str(data[example][feature]))
        example_data.append(eval(eval_string))
    composite_fp_matrix.append(example_data)

composite_fp_matrix = np.array(composite_fp_matrix)

'''
At this point composite_fp_matrix is a numerical representation of size m x n,
where m is the number of examples and n is the number of composite features

examples is a list of strings with length m, where m is the number of examples

composite_features is a list of strings with length n, where n is the number of
composite features. Each entry describes the operation used to calculate the
corresponding column in composite_fp_matrix

You should be able to check any value in the composite_fp_matrix[i][j] by
using the data from data[examples[i]] in the operation listed in
composite_features[j]

This approach is likely to fail if one feature label is a substring of another,
e.g. "radii" and "atomic_radii"

The methods here are quite general (I think) and could perhaps be added to some
portion of AtoML to be automated for users.
'''

print(composite_fp_matrix[2][23])
print(data[examples[2]])
print(composite_features[23])
