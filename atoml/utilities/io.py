"""Functions to read and write models to file."""
import pickle


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
