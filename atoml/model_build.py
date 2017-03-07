""" Functions to build a baseline model. """
import numpy as np

from .database_functions import DescriptorDatabase
from .fpm_operations import (get_order_2, get_order_2ab, get_ablog,
                             get_labels_order_2, get_labels_order_2ab,
                             get_labels_ablog)


def from_atoms(train_atoms, fpv_function, key, test_atoms=None,
               feature_names=None, db_name='fpv_store.sqlite'):
    """ Build model from a set of atoms objects. """
    train_matrix = fpv_function(train_atoms)
    if feature_names is None:
        feature_names = ['f' + str(i) for i in range(len(train_matrix[0]))]

    # Extend the feature matrix combinatorially.
    train_matrix = np.concatenate((train_matrix, get_order_2(train_matrix)),
                                  axis=1)
    train_matrix = np.concatenate((train_matrix, get_order_2ab(train_matrix)),
                                  axis=1)
    train_matrix = np.concatenate((train_matrix, get_ablog(train_matrix)),
                                  axis=1)

    # Extend the feature naming scheme.
    feature_names = feature_names + get_labels_order_2
    feature_names = feature_names + get_labels_order_2ab
    feature_names = feature_names + get_labels_ablog

    # Add on the id and target values.
    data_id = [a.info['unique_id'] for a in train_atoms]
    data_key = [a.info['key_value_pairs'][key] for a in train_atoms]
    train_dmat = np.concatenate((data_id, train_matrix, data_key), axis=1)

    # Define database parameters to store features.
    train_db = DescriptorDatabase(db_name='train_' + db_name,
                                  table='FingerVector')
    train_db.create_db(names=feature_names)
    train_db.fill_db(descriptor_names=feature_names, data=train_dmat)

    if test_atoms is not None:
        test_matrix = fpv_function(test_atoms)

        # Extend the feature matrix combinatorially.
        train_matrix = np.concatenate((test_matrix, get_order_2(test_matrix)),
                                      axis=1)
        train_matrix = np.concatenate((test_matrix,
                                       get_order_2ab(test_matrix)), axis=1)
        train_matrix = np.concatenate((test_matrix, get_ablog(test_matrix)),
                                      axis=1)

        # Add on the id and target values.
        data_id = [a.info['unique_id'] for a in test_atoms]
        data_key = [a.info['key_value_pairs'][key] for a in test_atoms]
        test_dmat = np.concatenate((data_id, test_matrix, data_key), axis=1)

        # Define database parameters to store features.
        train_db = DescriptorDatabase(db_name='test_' + db_name,
                                      table='FingerVector')
        train_db.create_db(names=feature_names)
        train_db.fill_db(descriptor_names=feature_names, data=test_dmat)


def from_matrix(train_matrix, test_matrix=None):
    """ Build a model from a pre-generated feature matrix. """
