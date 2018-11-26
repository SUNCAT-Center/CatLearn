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
        kernel_list = model.kernel_list
        write_train_data(
            filename, train_features, train_targets, regularization,
            kernel_list)
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
        train_features, train_targets, regularization, kernel_list = \
         read_train_data(filename)
        gp = GaussianProcess(
            train_fp=train_features, train_target=train_targets,
            kernel_list=kernel_list, regularization=regularization,
            optimize_hyperparameters=False)
        return gp
    else:
        raise NotImplementedError('{} file extension not implemented.'.format(
            ext))


def write_train_data(filename, train_features, train_targets, regularization,
                     kernel_list):
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
    kernel_list : dict
        The list containing dictionaries for the kernels.
    """
    f = h5py.File('{}.hdf5'.format(filename), 'w')
    f.create_dataset('train_features', data=train_features, compression='gzip',
                     compression_opts=9)
    f.create_dataset('train_targets', data=train_targets, compression='gzip',
                     compression_opts=9)
    f.create_dataset('regularization', data=regularization)
    _kernel_list_to_group(f, '/', kernel_list)


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
    kernel_list : list
        The dictionary containing parameters for the kernels.
    """
    f = h5py.File('{}.hdf5'.format(filename), 'r')
    train_features = np.asarray(f['train_features'])
    train_targets = np.asarray(f['train_targets'])
    regularization = float(np.asarray(f['regularization']))
    kernel_list = _load_kernel_list_from_group(f)

    return train_features, train_targets, regularization, kernel_list


def _kernel_list_to_group(h5file, path, klist):
    """Convert a list of dictionaries to group format.

    Parameters
    ----------
    h5file : hdf5
        An open hdf5 file object.
    path : str
        The path to write data in the hdf5 file object.
    klist : list
        List of dictionaries to save in hdf5 format.
    """
    for i, kdict in enumerate(klist):
        _dict_to_group(h5file, '/kernel_list/' + str(i) + '/', kdict)


def _load_kernel_list_from_group(h5file):
    """Convert a list of dictionaries to group format.

    Parameters
    ----------
    h5file : hdf5
        An open hdf5 file object.
    path : str
        The path to write data in the hdf5 file object.
    klist : list
        List of dictionaries to save in hdf5 format.
    
    Returns
    -----------
    kernel_list : list
        List of dictionaries for all the kernels.
    """
    h5file.keys()
    
    kernel_list = []
    for key, item in h5file['/kernel_list/'].items():
        kernel_list.append(_load_dict_from_group(h5file,
                                                 '/kernel_list/' + key + '/'))

    return kernel_list

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
