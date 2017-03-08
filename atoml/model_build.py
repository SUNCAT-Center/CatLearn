""" Functions to build a baseline model. """
import numpy as np

from .database_functions import DescriptorDatabase
from .fpm_operations import (get_order_2, get_order_2ab, get_ablog,
                             get_div_order_2, get_labels_order_2,
                             get_labels_order_2ab, get_labels_ablog)


def from_atoms(train_atoms, fpv_function, train_target, test_atoms=None,
               test_target=None, feature_names=None, create_db=True,
               db_name='fpv_store.sqlite'):
    """ Build model from a set of atoms objects. """
    train_matrix = fpv_function(train_atoms)
    if feature_names is None:
        feature_names = ['f' + str(i) for i in range(len(train_matrix[0]))]
    train_id = [a.info['unique_id'] for a in train_atoms]

    if test_atoms is not None:
        test_matrix = fpv_function(test_atoms)
        test_id = [a.info['unique_id'] for a in test_atoms]
        return from_matrix(train_matrix=train_matrix,
                           feature_names=feature_names, train_id=train_id,
                           train_target=train_target, test_matrix=test_matrix,
                           test_id=test_id, test_target=test_target,
                           create_db=create_db, db_name=db_name)
    else:
        return from_matrix(train_matrix=train_matrix,
                           feature_names=feature_names, train_id=train_id,
                           train_target=train_target, create_db=create_db,
                           db_name=db_name)


def from_matrix(train_matrix, feature_names, train_id, train_target,
                test_matrix=None, test_id=None, test_target=None,
                create_db=True, db_name='fpv_store.sqlite'):
    """ Build a model from a pre-generated feature matrix. """
    train_matrix, feature_names = expand_matrix(train_matrix, feature_names)
    # Add on the id and target values.
    train_id = [[i] for i in train_id]
    train_dmat = np.append(train_id, train_matrix, axis=1)
    train_target = [[i] for i in train_target]
    train_dmat = np.append(train_dmat, train_target, axis=1)

    dnames = feature_names + ['target']

    if create_db:
        # Define database parameters to store features.
        train_db = DescriptorDatabase(db_name='train_' + db_name,
                                      table='FingerVector')
        train_db.create_db(names=dnames)
        train_db.fill_db(descriptor_names=dnames, data=train_dmat)

    if test_matrix is not None:
        test_matrix = expand_matrix(test_matrix, return_names=False)
        # Add on the id and target values.
        test_id = [[i] for i in test_id]
        test_dmat = np.append(test_id, test_matrix, axis=1)
        test_target = [[i] for i in test_target]
        test_dmat = np.append(test_dmat, test_target, axis=1)

        if create_db:
            # Define database parameters to store features.
            test_db = DescriptorDatabase(db_name='test_' + db_name,
                                         table='FingerVector')
            test_db.create_db(names=dnames)
            test_db.fill_db(descriptor_names=dnames, data=test_dmat)


def expand_matrix(feature_matrix, feature_names=None, return_names=True):
    # Extend the feature matrix combinatorially.
    order_2 = get_order_2(feature_matrix)
    div_order_2 = get_div_order_2(feature_matrix)
    order_2ab = get_order_2ab(feature_matrix, a=2, b=3)
    ablog = get_ablog(feature_matrix, a=2, b=3)

    feature_matrix = np.concatenate((feature_matrix, order_2, div_order_2,
                                     order_2ab, ablog), axis=1)

    # Extend the feature naming scheme.
    if return_names:
        order_2 = get_labels_order_2(feature_names)
        div_order_2 = get_labels_order_2(feature_names, div=True)
        order_2ab = get_labels_order_2ab(feature_names, a=2, b=3)
        ablog = get_labels_ablog(feature_names, a=2, b=3)
        feature_names += order_2 + div_order_2 + order_2ab + ablog

        return feature_matrix, feature_names

    else:
        return feature_matrix
