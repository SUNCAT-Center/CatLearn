"""Functions to read and write models to file."""
import pickle
import h5py


def write(filename, model):
    """Function to write a pickle of model object.

    Parameters
    ----------
    filename : str
        The name of the save file.
    model : obj
        Python GaussianProcess object.
    """
    with open('{}.pkl'.format(filename), 'wb') as outfile:
        pickle.dump(model, outfile, pickle.HIGHEST_PROTOCOL)


def read(filename):
    """Function to read a pickle of model object.

    Parameters
    ----------
    filename : str
        The name of the save file.

    Returns
    -------
    model : obj
        Python GaussianProcess object.
    """
    with open('{}.pkl'.format(filename), 'rb') as infile:
        return pickle.load(infile)


def write_train_data(filename, train_features, train_targets):
    """Function to test raw data save.

    Parameters
    ----------
    filename : str
        The name of the save file.
    """
    f = h5py.File('{}.hdf5'.format(filename), 'w')
    f.create_dataset('train_features', data=train_features, compression='gzip',
                     compression_opts=9)
    f.create_dataset('train_targets', data=train_targets, compression='gzip',
                     compression_opts=9)
