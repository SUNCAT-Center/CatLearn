""" Functions to build a baseline model. """
import numpy as np
from math import log

from .database_functions import DescriptorDatabase
from .fpm_operations import (get_order_2, get_order_2ab, get_ablog,
                             get_div_order_2, get_labels_order_2,
                             get_labels_order_2ab, get_labels_ablog)
from .feature_select import iterative_screening


class ModelBuilder(object):
    def __init__(self, create_db=True, db_name='fpv_store.sqlite',
                 screening_method='rrcs', screening_correlation='kendall'):
        self.create_db = create_db
        self.db_name = db_name
        self.screening_method = screening_method
        self.screening_correlation = screening_correlation

    def from_atoms(self, train_atoms, fpv_function, train_target,
                   test_atoms=None, test_target=None, feature_names=None):
        """ Build model from a set of atoms objects. """
        train_matrix = fpv_function(train_atoms)
        train_id = [a.info['unique_id'] for a in train_atoms]

        # Generate standard feature neames for basic tracking.
        if feature_names is None:
            feature_names = ['f' + str(i) for i in
                             range(len(train_matrix[0]))]

        # Make feature matrix for test data if atoms objects are supplied.
        test_id, test_matrix = None, None
        if test_atoms is not None:
            test_matrix = fpv_function(test_atoms)
            test_id = [a.info['unique_id'] for a in test_atoms]

        return self.from_matrix(train_matrix=train_matrix,
                                feature_names=feature_names,
                                train_id=train_id,
                                train_target=train_target,
                                test_matrix=test_matrix, test_id=test_id,
                                test_target=test_target)

    def from_matrix(self, train_matrix, feature_names, train_id,
                    train_target, test_matrix=None, test_id=None,
                    test_target=None):
        """ Build a model from a pre-generated feature matrix. """
        train_matrix, feature_names = self.expand_matrix(train_matrix,
                                                         feature_names)
        if self.create_db:
            self.db_store(type='train', atoms_id=train_id,
                          feature_matrix=train_matrix, target=train_target,
                          feature_names=feature_names)

        if test_matrix is not None:
            test_matrix = self.expand_matrix(test_matrix,
                                             return_names=False)
            if self.create_db:
                self.db_store(type='test', atoms_id=test_id,
                              feature_matrix=test_matrix,
                              target=test_target,
                              feature_names=feature_names)

        redm = self.reduce_matrix(train_matrix=train_matrix,
                                  test_matrix=test_matrix,
                                  target=train_target,
                                  feature_names=feature_names)

    def expand_matrix(self, feature_matrix, feature_names=None,
                      return_names=True):
        """ Expand the feature matrix by combing original features. """
        # Extend the feature matrix combinatorially.
        order_2 = get_order_2(feature_matrix)
        div_order_2 = get_div_order_2(feature_matrix)
        order_2ab = get_order_2ab(feature_matrix, a=2, b=3)
        ablog = get_ablog(feature_matrix, a=2, b=3)

        feature_matrix = np.concatenate((feature_matrix, order_2,
                                         div_order_2, order_2ab, ablog),
                                        axis=1)

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

    def reduce_matrix(self, train_matrix, target, feature_names,
                      test_matrix=None):
        """ Function to reduce the feature space. """
        # Check to see if there are more features than data points.
        d = len(train_matrix[1])
        n = len(train_matrix)
        if d > n:
            # Use correlation screening to reduce features down to number
            # of data.
            s = int(round(log(d/n)**0.5, 0))
            if s == 0:
                s = 1
            itred = iterative_screening(target=target, train_fpv=train_matrix,
                                        test_fpv=test_matrix, size=n, step=s,
                                        method=self.screening_method,
                                        corr=self.screening_correlation)
            # Update the feature matrix.
            train_matrix = itred['train_fpv']
            if test_matrix is not None:
                test_matrix = itred['test_fpv']

        # Ridge regression to get ordering of features.

        # LOOCV testing.

    def db_store(self, type, atoms_id, feature_matrix, target,
                 feature_names):
        """ Function to automatically store feature matrix. """
        # Add on the id and target values.
        atoms_id = [[i] for i in atoms_id]
        dmat = np.append(atoms_id, feature_matrix, axis=1)
        target = [[i] for i in target]
        dmat = np.append(dmat, target, axis=1)

        dnames = feature_names + ['target']

        # Define database parameters to store features.
        new_db = DescriptorDatabase(db_name=type + '_' + self.db_name,
                                    table='FingerVector')
        new_db.create_db(names=dnames)
        new_db.fill_db(descriptor_names=dnames, data=dmat)
