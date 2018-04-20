"""Functions to read and write models to file."""
import pickle
import h5py
import numpy as np

from catlearn.regression import GaussianProcess


def write(filename, model, ext='pkl'):
    """Function to write a pickle of model object.

    Parameters
    ----------
    filename : str
        The name of the save file.
    model : obj
        Python GaussianProcess object.
    ext : str
        Format to save GP, can be pkl or hdf5. Default is pkl.
    """
    if ext is 'pkl':
        with open('{}.pkl'.format(filename), 'wb') as outfile:
            pickle.dump(model, outfile, pickle.HIGHEST_PROTOCOL)
    elif ext is 'hdf5':
        train_features = model.train_fp
        train_targets = model.train_target
        regularization = model.regularization
        kernel_dict = model.kernel_dict
        write_train_data(
            filename, train_features, train_targets, regularization,
            kernel_dict)
    else:
        raise NotImplementedError('{} file extension not implemented.'.format(
            ext))


def read(filename, ext='pkl'):
    """Function to read a pickle of model object.

    Parameters
    ----------
    filename : str
        The name of the save file.
    ext : str
        Format to save GP, can be pkl or hdf5. Default is pkl.

    Returns
    -------
    model : obj
        Python GaussianProcess object.
    """
    if ext is 'pkl':
        with open('{}.pkl'.format(filename), 'rb') as infile:
            return pickle.load(infile)
    elif ext is 'hdf5':
        train_features, train_targets, regularization, kernel_dict = \
         read_train_data(filename)
        gp = GaussianProcess(
            train_fp=train_features, train_target=train_targets,
            kernel_dict=kernel_dict, regularization=regularization,
            optimize_hyperparameters=False)
        return gp
    else:
        raise NotImplementedError('{} file extension not implemented.'.format(
            ext))


def write_train_data(filename, train_features, train_targets, regularization,
                     kernel_dict):
    """Function to write raw training data.

    Parameters
    ----------
    filename : str
        The name of the save file.
    train_features : arr
        Arry of the training features.
    train_targets : list
        A list of the training targets.
    regularization : float
        The regularization parameter.
    kernel_dict : dict
        The dictionary containing parameters for the kernels.
    """
    f = h5py.File('{}.hdf5'.format(filename), 'w')
    f.create_dataset('train_features', data=train_features, compression='gzip',
                     compression_opts=9)
    f.create_dataset('train_targets', data=train_targets, compression='gzip',
                     compression_opts=9)
    f.create_dataset('regularization', data=regularization)
    _dict_to_group(f, '/', kernel_dict)


def read_train_data(filename):
    """Function to read raw training data.

    Parameters
    ----------
    filename : str
        The name of the save file.

    Returns
    -------
    train_features : arr
        Arry of the training features.
    train_targets : list
        A list of the training targets.
    regularization : float
        The regularization parameter.
    kernel_dict : dict
        The dictionary containing parameters for the kernels.
    """
    f = h5py.File('{}.hdf5'.format(filename), 'r')
    train_features = np.asarray(f['train_features'])
    train_targets = np.asarray(f['train_targets'])
    regularization = float(np.asarray(f['regularization']))
    kernel_dict = _load_dict_from_group(f, '/')

    return train_features, train_targets, regularization, kernel_dict


def _dict_to_group(h5file, path, sdict):
    """Convert dictionary format to group format.

    Parameters
    ----------
    h5file : hdf5
        An open hdf5 file object.
    path : str
        The path to write data in the hdf5 file object.
    sdict : dict
        Dictionary to save in hdf5 format.
    """
    for key, item in sdict.items():
        if isinstance(item,
                      (np.ndarray, np.int64, np.float64, str, float, list)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            _dict_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type' % type(item))


def _load_dict_from_group(h5file, path):
    """Convert group format to dictionary format.

    Parameters
    ----------
    h5file : hdf5
        An open hdf5 file object.
    path : str
        The path to load data from the hdf5 file object.

    Returns
    -------
    rdict : dict
        The resulting dictionary.
    """
    rdict = {}
    for key, item in h5file[path].items():
        if key != 'train_features' and key != 'train_targets' and \
         key != 'regularization':
            if isinstance(item, h5py._hl.dataset.Dataset):
                rdict[key] = item.value
            elif isinstance(item, h5py._hl.group.Group):
                rdict[key] = _load_dict_from_group(h5file, path + key + '/')

    return rdict
