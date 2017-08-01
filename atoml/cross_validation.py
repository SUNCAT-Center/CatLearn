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

    def global_scale_data(self, index_split, scale='standardize'):
        """Make an array with all data.

        Parameters
        ----------
        index_split : array
            Array with the index data.
        scale : string
            Method of scaling, can be either 'standardize' or 'normalize'.
        """
        global_data1 = self._compile_split(index_split['1_1'])
        global_data2 = self._compile_split(index_split['1_2'])
        global_feat1 = np.array(global_data1[:, 1:-1], np.float64)
        global_feat2 = np.array(global_data2[:, 1:-1], np.float64)
        globaldata = np.concatenate((global_feat1, global_feat2), axis=0)

        if scale is 'standardize':
            s = standardize(train_matrix=globaldata)
            mean, scalar = s['mean'], s['std']
        if scale is 'normalize':
            s = normalize(train_matrix=globaldata)
            mean, scalar = s['mean'], s['dif']

        return mean, scalar

    def get_train_data(self, index_split, indicies):
        """Make array with training data according to index.

        Parameters
        ----------
        index_split : array
            Array with the index data.
        indicies : array
            Index used to generate data.
        """
        index1, index2 = indicies.split('_')

        train_data = self._compile_split(index_split[index1 + '_' + index2])

        train_features = np.array(train_data[:, 1:-1], np.float64)
        train_targets = np.array(train_data[:, -1:], np.float64)

        d1, d2 = np.shape(train_targets)
        train_targets = train_targets.reshape(d2, d1)[0]

        return train_targets, train_features, index1, index2

    def get_test_data(self, index_split, index1, split):
        """Make array with test data according to data.

        Parameters
        ----------
        index_split : array
            Array with the index data.
        index1 : int
            The first number in the data index.
        split : int
            The second number in the data index.
        """
        test_data = self._compile_split(index_split[index1 + '_' +
                                                    str(split)])
        test_features = np.array(test_data[:, 1:-1], np.float64)
        test_targets = np.array(test_data[:, -1:], np.float64)
        d1, d2 = np.shape(test_targets)
        test_targets = test_targets.reshape(d2, d1)[0]

        return test_features, test_targets

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
