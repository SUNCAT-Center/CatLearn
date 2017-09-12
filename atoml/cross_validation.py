"""Cross validation routines to work with feature database."""
import sqlite3
import json
import yaml
import pickle
import numpy as np
from random import shuffle
from collections import OrderedDict

from atoml.feature_preprocess import standardize, normalize


class HierarchyValidation(object):
    """Class to form hierarchy crossvalidation setup.

    This class is used to cross-validate with respect to data size. The initial
    dataset is split in two and subsequent datasets split further until a
    minimum size is reached. Predictions are made on all subsets of data giving
    averaged error and certainty at each data size.
    """

    def __init__(self, file_name, db_name, table, file_format='pickle'):
        """Hierarchy crossvalidation setup.

        Parameters
        ----------
        file_name : string
            Name of file to store the row id for the substes of data. Do not
            append format type.
        db_name : string
            Database name.
        table : string
            Name of the table in database.
        file_format : string
            Format to save the splitting data, either json, yaml or pickle
            type. Default is binary pickle file.
        """
        self.file_name = file_name
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.table = table
        self.file_format = file_format

    def split_index(self, min_split, max_split=None, all_index=None):
        """Function to split up the db index to form subsets of data.

        Parameters
        ----------
        min_split : int
            Minimum size of a data subset.
        max_split : int
            Maximum size of a data subset.
        all_index : list
            List of indices in the feature database.
        """
        data = OrderedDict()
        if all_index is None:
            all_index = self._get_index()

        # Randomize the indices and remove any remainder from first split.
        shuffle(all_index)
        if max_split is not None:
            assert len(all_index) > max_split and max_split >= 2 * min_split
            # Cut off the list of indices.
            all_index = all_index[:max_split]

        assert len(all_index) > min_split
        size = int(len(all_index)/2)
        data['1_1'], data['1_2'] = all_index[:size], all_index[size:]

        # TODO fix no_split because it is way too large.
        no_split = int(min(len(data['1_1']), len(data['1_2'])) / min_split)

        for i in range(1, no_split+1):
            subsplit = 2 ** i
            sn = 1
            for j in range(1, subsplit+1):
                current_split = data[str(i) + '_' + str(j)]
                shuffle(current_split)
                new_split = int(len(current_split) / 2)
                if new_split >= min_split:
                    first_name, sn = str(i+1) + '_' + str(sn), sn + 1
                    second_name, sn = str(i+1) + '_' + str(sn), sn + 1
                    data[first_name] = current_split[:new_split]
                    data[second_name] = current_split[new_split:]
                else:
                    self._write_split(data)
                    return data

    def load_split(self):
        """Function to load the split from file."""
        if self.file_format is not 'pickle':
            with open(self.file_name + '.' +
                      self.file_format, 'r') as textfile:
                if self.file_format is 'json':
                    data = json.load(textfile)
                if self.file_format is 'yaml':
                    data = yaml.load(textfile)
        else:
            with open(self.file_name + '.' +
                      self.file_format, 'rb') as textfile:
                data = pickle.load(textfile)

        return data

    def global_scale_data(self, index_split, scale='standardize',
                          features=None,):
        """Find scaling for all available data.

        Parameters
        ----------
        index_split : array
            Array with the index data.
        scale : string
            Method of scaling, can be either 'standardize' or 'normalize'.
        """
        if features is None:
            features = -1
        data1 = self._compile_split(index_split['1_1'])
        data2 = self._compile_split(index_split['1_2'])
        feat1 = np.array(data1[:, 1:features], np.float64)
        feat2 = np.array(data2[:, 1:features], np.float64)

        if scale is 'standardize':
            s = standardize(train_matrix=feat1, test_matrix=feat2, local=False)
            mean, scalar = s['mean'], s['std']
        if scale is 'normalize':
            s = normalize(train_matrix=feat1, test_matrix=feat2, local=False)
            mean, scalar = s['mean'], s['dif']

        return scalar, mean

    def get_subset_data(self, index_split, indicies, split=None):
        """Make array with training data according to index.

        Parameters
        ----------
        index_split : array
            Array with the index data.
        indicies : array
        Index used to generate data.
        """
        index1, index2 = indicies.split('_')
        if split is None:
            data = self._compile_split(index_split[index1 + '_' + index2])
        else:
            data = self._compile_split(index_split[index1 + '_' + str(split)])
        data_features = np.array(data[:, 1:-1], np.float64)
        data_targets = np.array(data[:, -1:], np.float64)

        d1, d2 = np.shape(data_targets)
        data_targets = data_targets.reshape(d2, d1)[0]

        return data_targets, data_features, index1, index2

    def _get_index(self):
        """Function to get the list of possible indices."""
        data = []
        for row in self.cursor.execute("SELECT uuid FROM %(table)s"
                                       % {'table': self.table}):
            data.append(row[0])

        return data

    def _write_split(self, data):
        """Function to write the split to file.

        Parameters
        ----------
        data : dict
            Index dict generated within the split_index function.
        """
        if self.file_format is not 'pickle':
            with open(self.file_name + '.' +
                      self.file_format, 'w') as textfile:
                if self.file_format is 'json':
                    json.dump(data, textfile)
                if self.file_format is 'yaml':
                    yaml.dump(data, textfile)
        else:
            with open(self.file_name + '.' +
                      self.file_format, 'wb') as textfile:
                pickle.dump(data, textfile, protocol=pickle.HIGHEST_PROTOCOL)

    def _compile_split(self, id_list):
        """Function to get actual data from database.

        Parameters
        ----------
        id_list : list
            The uuids to pull data.
        """
        if len(id_list) > 999:
            store_data = self._get_data(id_list[:999])
            for i in range(1, int(len(id_list) / 999)+1):
                start_index = i * 999
                if len(id_list[start_index:]) < 999:
                    more_data = self._get_data(id_list[start_index:])
                else:
                    more_data = self._get_data(id_list[start_index:(i+1)*999])
                store_data = np.concatenate((store_data, more_data), axis=0)
        else:
            store_data = np.asarray(self._get_data(id_list))

        assert np.shape(store_data)[0] == len(id_list)

        return store_data

    def _get_data(self, id_list):
        """Function to extract raw data from the database.

        Parameters
        ----------
        id_list : list
            The uuids to pull data.
        """
        qu = ','.join('?' for i in id_list)
        query = 'SELECT * FROM %(table)s WHERE uuid IN (%(uid)s)' \
            % {'table': self.table, 'uid': qu}
        self.cursor.execute(query, id_list)

        return self.cursor.fetchall()

    def hierarcy(self, features, min_split, max_split, new_data=True,
                 ridge=True, scale=True, globalscale=True, normalization=True,
                 feature_selection=False):
        """Function to extract raw data from the database.

        Parameters
        ----------
        id_list : list
            The uuids to pull data.
        """
        from data_process import data_process
        from feature_selection import feature_selection
        from atoml.ridge_regression import RidgeRegression

        result = []
        data_size = []
        p_error = []
        PC = data_process(features, min_split, max_split, scale=scale,
                          ridge=ridge, normalization=normalization)

        if new_data:
            # Split the data into subsets.
            self.split_index(min_split=min_split, max_split=max_split)
            # Load data back in from save file.
        index_split = self.load_split()

        if globalscale:
            s_feat, m_feat = self.global_scale_data(index_split, features)
        for indicies in reversed(index_split):
            if not globalscale:
                s_feat, m_feat = None, None

            s_tar, m_tar = None, None
            train_targets, train_features, index1, index2 =\
                self.get_subset_data(index_split=index_split,
                                     indicies=indicies)
            coef = None

            for split in range(1, 2**int(index1)+1):
                reg_data = {'result': None}
                if split != int(index2):
                    _, train_features, _, _ = self.get_subset_data(
                                                    index_split=index_split,
                                                    indicies=indicies)
                    test_targets, test_features, _, _ =\
                        self.get_subset_data(index_split, indicies,
                                             split=split)
                    ridge = RidgeRegression()
                    (s_tar, m_tar, s_feat, m_feat,
                     train_targets, train_features,
                     test_features) = \
                        PC.scaling_data(train_features=train_features,
                                        train_targets=train_targets,
                                        test_features=test_features,
                                        s_tar=s_tar, m_tar=m_tar,
                                        s_feat=s_feat, m_feat=m_feat)
                    #if feature_selection:
                        #FS = feature_selection(train_features=train_features,
                                               #train_targets=train_targets)
                        #selected_features = FS.selection()
                    if coef is None:
                        reg_data = ridge.regularization(train_targets,
                                                        train_features,
                                                        coef)

                    if reg_data['result'] is not None:
                        reg_store = reg_data['result']
                        coef = reg_data['result'][0]

                    data = PC.prediction_error(test_features, test_targets,
                                               coef, s_tar, m_tar)

                    if reg_data['result'] is not None:

                        data['result'] += reg_data['result']

                    else:

                        data['result'] += reg_store

                    result.append(data['result'])
                    print('data size:', data['result'][0], 'prediction error:',
                          data['result'][1], 'Omega:', data['result'][5],
                          'Euclidean length:', data['result'][2],
                          'Pearson correlation:', data['result'][3])

                    data_size.append(data['result'][0])
                    p_error.append(data['result'][1])

        p_error_mean_list, data_size_mean_list, corrected_std =\
            PC.get_statistic(data_size, p_error)
        PC.learning_curve(data_size, p_error,
                          data_size_mean=data_size_mean_list,
                          p_error_mean=p_error_mean_list,
                          corrected_std=corrected_std)
