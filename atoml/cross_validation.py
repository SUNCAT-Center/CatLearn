"""Cross validation routines to work with feature database."""
import sqlite3
import json
import yaml
import pickle
import numpy as np
from random import shuffle


class HierarchyValidation(object):
    """Class to form hierarchy crossvalidation setup.

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

    def __init__(self, file_name, db_name, table, file_format='pickle'):
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
        data = {}
        if all_index is None:
            all_index = self._get_index()

        # Randomize the indices and remove any remainder from first split.
        shuffle(all_index)

        if max_split is not None:
            assert len(all_index) > max_split
            # Cut off the list of indices.
            all_index = all_index[:max_split]

        assert len(all_index) > min_split
        size = int(len(all_index)/2)
        data['1_1'], data['1_2'] = all_index[:size], all_index[size:]

        # TODO fix no_split because it is way too large.
        no_split = int(min(len(data['1_1']), len(data['1_2'])) / min_split)

        for i in range(1, no_split+1):
            subsplit = i * 2
            sn = 1
            for j in range(1, subsplit+1):
                current_split = data[str(i) + '_' + str(j)]
                shuffle(current_split)
                new_split = int(len(current_split) / 2)
                if new_split > min_split:
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

    def split_predict(self, index_split, predict):
        """Function to make predictions looping over all subsets of data.

        Parameters
        ----------
        index_split : dict
            All data for the split.
        predict : function
            The prediction function. Must return dict with 'result' in it.
        """
        result = []
        for i in index_split:
            j, k = i.split('_')
            train_data = self._compile_split(index_split[j + '_' + k])
            train_features = np.array(train_data[:, 1:-1], np.float64)
            train_targets = np.array(train_data[:, -1:], np.float64)
            d1, d2 = np.shape(train_targets)
            train_targets = train_targets.reshape(d2, d1)[0]

            if int(k) % 2 == 1:
                test_data = self._compile_split(index_split[j + '_' +
                                                            str(int(k)+1)])
            else:
                test_data = self._compile_split(index_split[j + '_' +
                                                            str(int(k)-1)])
            test_features = np.array(test_data[:, 1:-1], np.float64)
            test_targets = np.array(test_data[:, -1:], np.float64)
            d1, d2 = np.shape(test_targets)
            test_targets = test_targets.reshape(d2, d1)[0]

            pred = predict(train_features=train_features,
                           train_targets=train_targets,
                           test_features=test_features,
                           test_targets=test_targets)
            result.append(pred['result'])

        return result

    def _get_index(self):
        """Function to get the list of possible indices."""
        data = []
        for row in self.cursor.execute("SELECT uuid FROM %(table)s"
                                       % {'table': self.table}):
            data.append(row[0])

        return data

    def _write_split(self, data):
        """Function to write the split to file."""
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
