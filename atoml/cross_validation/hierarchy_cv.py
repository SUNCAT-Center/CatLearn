"""Cross validation routines to work with feature database."""
import sqlite3
import json
import yaml
import pickle
import numpy as np
from random import shuffle
from collections import OrderedDict
import uuid

from atoml.utilities import DescriptorDatabase


class Hierarchy(object):
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
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        self.table = table
        self.file_format = file_format

    def todb(self, features, targets):
        """Function to convert numpy arrays to basic db."""
        data = np.concatenate(
            (features, np.reshape(targets, (len(targets), 1))), axis=1)
        uid = [str(uuid.uuid4()) for _ in range(len(targets))]
        data = np.concatenate((np.reshape(uid, (len(uid), 1)), data), axis=1)

        descriptors = ['f' + str(i) for i in range(np.shape(features)[1])]
        targets = ['target']
        names = descriptors + targets

        # Set up the database to save system descriptors.
        dd = DescriptorDatabase(db_name=self.db_name, table=self.table)
        dd.create_db(names=names)

        # Fill the database with the data.
        dd.fill_db(descriptor_names=names, data=data)

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
        size = int(len(all_index) / 2)
        data['1_1'], data['1_2'] = all_index[:size], all_index[size:]

        # TODO fix no_split because it is way too large.
        no_split = int(min(len(data['1_1']), len(data['1_2'])) / min_split)

        for i in range(1, no_split + 1):
            subsplit = 2 ** i
            sn = 1
            for j in range(1, subsplit + 1):
                current_split = data[str(i) + '_' + str(j)]
                shuffle(current_split)
                new_split = int(len(current_split) / 2)
                if new_split >= min_split:
                    first_name, sn = str(i + 1) + '_' + str(sn), sn + 1
                    second_name, sn = str(i + 1) + '_' + str(sn), sn + 1
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
            for i in range(1, int(len(id_list) / 999) + 1):
                start_index = i * 999
                if len(id_list[start_index:]) < 999:
                    more_data = self._get_data(id_list[start_index:])
                else:
                    more_data = self._get_data(
                        id_list[start_index:(i + 1) * 999])
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

    def split_predict(self, index_split, predict, **kwargs):
        """Function to make predictions looping over all subsets of data.

        Parameters
        ----------
        index_split : dict
            All data for the split.
        predict : function
            The prediction function. Must return dict with 'result' in it.

        """
        result = []
        size = []
        for i in reversed(index_split):
            j, k = i.split('_')
            train_data = self._compile_split(index_split[i])
            train_features = np.array(train_data[:, 1:-1], np.float64)
            train_targets = np.array(train_data[:, -1:], np.float64)
            d1, d2 = np.shape(train_targets)
            train_targets = train_targets.reshape(d2, d1)[0]

            for m in reversed(index_split):
                n, o = m.split('_')
                if n == j:
                    if k != o:
                        test_data = self._compile_split(
                            index_split[m])

                        test_features = np.array(test_data[:, 1:-1],
                                                 np.float64)
                        test_targets = np.array(test_data[:, -1:], np.float64)
                        d1, d2 = np.shape(test_targets)
                        test_targets = test_targets.reshape(d2, d1)[0]

                        pred = predict(train_features=train_features,
                                       train_targets=train_targets,
                                       test_features=test_features,
                                       test_targets=test_targets, **kwargs)

                        if 'size' in pred:
                            size.append(pred['size'])
                        result.append(pred['result'])

        return result, size
