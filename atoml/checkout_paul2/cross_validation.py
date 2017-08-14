"""Cross validation routines to work with feature database."""
import sqlite3
import json
import yaml
import pickle
import numpy as np
from random import shuffle
from collections import OrderedDict
from data_process import data_process
from stream import placeholder
from predict import target_normalize


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

    def globalscaledata(self, index_split):
        """Make an array with all data.

        Parameters
        ----------
        index_split : array
            Array with the index data.
        """
        global_data1 = self._compile_split(index_split["1" + '_' + "1"])
        global_data2 = self._compile_split(index_split["1" + '_' + "2"])
        global_feat1 = np.array(global_data1[:, 1:-1], np.float64)
        global_tar1 = np.array(global_data1[:, -1:], np.float64)
        d1, d2 = np.shape(global_tar1)
        global_tar1 = global_tar1.reshape(d2, d1)[0]
        global_feat2 = np.array(global_data2[:, 1:-1], np.float64)
        global_tar2 = np.array(global_data2[:, -1:], np.float64)
        d1, d2 = np.shape(global_tar2)
        global_tar2 = global_tar2.reshape(d2, d1)[0]
        globaldata = np.concatenate((global_feat1, global_feat2), axis=0)
        return globaldata, global_feat1, global_tar1

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

    def hierarcy(self, features, min_split, max_split, hv, new_data=True,
                 ridge=True, scale=True, globalscale=True, normalization=True,
                 featselect_featvar=False, featselect_featconst=True,
                 select_limit=None, feat_sub=15):
        """Function to extract raw data from the database.

        Parameters
        ----------
        features : int
            Number of features used for regression.
        min_split : int
            Number of datasplit in the smallest sub-set.
        max_split : int
            Number of datasplit in the largest sub-set.
        new_data : string
           Use new data or the previous data.
        ridge : string
            Ridge regulazer is deafult. If False, lasso is used.
        scale : string
            If the data are supposed to be scaled or not.
        globalscale : string
            Using global scaleing or not.
        normalization : string
            If scaled, normalized or standardized. Normalized is default.
        feature_selection : string
           Using feature selection with ridge, or plain vanilla ridge.
        select_limit : int
           Up to have many number of features used for feature selection.
        """
        result, set_size, p_error = [], [], [],
        hier_level = int(np.log(max_split/min_split)/np.log(2))
        PC = data_process(features, min_split, max_split, scale=scale,
                          ridge=ridge, normalization=normalization)
        selected_features = None
        if new_data:
            # Split the data into subsets.
            self.split_index(min_split, max_split=max_split)
            # Load data back in from save file.
        index_split = self.load_split()
        if globalscale:
            # Get all the data, and one of the largest sub-set.
            globalscaledata, glob_feat1, glob_tar1 = self.globalscaledata(
                                                     index_split)
            # Statistics for global scaling, and scales largest sub-set.
            s_feat, m_feat, glob_feat1 = PC.globalscaling(globalscaledata,
                                                          glob_feat1)
            data = target_normalize(glob_tar1)
            glob_tar1 = data['target']
        else:
            s_feat, m_feat = None, None

        for indicies in reversed(index_split):
            ph = placeholder(globalscale, PC, index_split, hv,
                             indicies, hier_level, featselect_featvar,
                             featselect_featconst, s_feat, m_feat,
                             select_limit=select_limit,
                             selected_features=selected_features,
                             feat_sub=feat_sub, glob_feat1=glob_feat1,
                             glob_tar1=glob_tar1)
            (set_size, p_error, result,
             index2, selected_features) = ph.predict_subsets(
                                      set_size=set_size,
                                      p_error=p_error,
                                      result=[])
            if int(index2) == 1:
                PC.featselect_featvar_plot(p_error, set_size)
            if (set_size and p_error) is None:
                return set_size, p_error, PC
        if not featselect_featvar:
            return set_size, p_error, PC
        else:
            exit()
