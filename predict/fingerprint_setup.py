import numpy as np
from collections import defaultdict


def get_single_fpv(candidates, fpv_name):
    """ Function to return the array of fingerprint vectors from a list of
        atoms objects.
    """
    # Check to see if we are dealing with a list of candidates or a single
    # atoms object.
    if type(candidates) is defaultdict or type(candidates) is list:
        list_fp = []
        for c in candidates:
            list_fp.append(get_fpv(c, fpv_name))
        return np.array(list_fp)
    # Do the same but for a single atoms object.
    else:
        c = candidates
        return np.array([get_fpv(c, fpv_name)])


def get_fpv(c, fpv_name):
    """ Get the fingerprint vector as an array from a single Atoms object.
        If a fingerprint vector is saved in info['data']['fpv'] it is returned
        otherwise saved in the data dictionary.
    """
    if 'data' not in c.info:
        c.info['data'] = {'fpv': fpv_name(atoms=c)}
    elif 'fpv' not in c.info['data']:
        c.info['data']['fpv'] = fpv_name(atoms=c)
    return c.info['data']['fpv']


def get_combined_fpv(candidates, fpv_list):
    """ Function to sequentially combine fingerprint vectors and return them
        for a list of atoms objects.
    """
    # Check that there are at least two fingerprint descriptors to combine.
    msg = "This functions combines various fingerprint"
    msg += " vectors, there must be at least two to combine"
    assert len(fpv_list) >= 2, msg

    # Check to see if we are dealing with a list of candidates or a single
    # atoms object.
    if type(candidates) is defaultdict or type(candidates) is list:
        list_fp = []
        for c in candidates:
            if 'data' in c.info and 'fpv' in c.info['data']:
                list_fp.append(c.info['data']['fpv'])
            else:
                fpv = fpv_list[0](atoms=c)
                for i in fpv_list[1:]:
                    fpv = np.concatenate((i(atoms=c), fpv))
                list_fp.append(fpv)

        return np.array(list_fp)
    # Do the same but for a single atoms object.
    else:
        c = candidates
        if 'data' in c.info and 'fpv' in c.info['data']:
            return np.array([c.info['data']['fpv']])
        else:
            fpv = fpv_list[0](atoms=c)
            for i in fpv_list[1:]:
                fpv = np.concatenate((i(atoms=c), fpv))
            return np.array([fpv])


def standardize(train, test=None):
    """ Standardize each descriptor in the FPV relative to the mean and
        standard deviation. If test data is supplied it is standardized
        relative to the training dataset.

        train: list
            List of atoms objects to be used as training dataset.

        test: list
            List of atoms objects to be used as test dataset.
    """
    std_fpv = []
    mean_fpv = []
    tt = np.transpose(train)
    for i in range(len(tt)):
        std_fpv.append(float(np.std(tt[i])))
        mean_fpv.append(float(np.mean(tt[i])))

    # Replace zero std with value 1 for devision.
    std_fpv = np.array(std_fpv)
    np.place(std_fpv, std_fpv == 0., [1.])

    std = defaultdict(list)
    for i in train:
        std['train'].append((i - mean_fpv) / std_fpv)
    if test is not None:
        for i in test:
            std['test'].append((i - mean_fpv) / std_fpv)
    std['std'] = std_fpv
    std['mean'] = mean_fpv

    return std


def normalize(train, test=None):
    """ Normalize each descriptor in the FPV to min/max or mean centered. If
        test data is supplied it is standardized relative to the training
        dataset.
    """
    max_fpv = []
    min_fpv = []
    mean_fpv = []
    tt = np.transpose(train)
    for i in range(len(tt)):
        max_fpv.append(float(max(tt[i])))
        min_fpv.append(float(min(tt[i])))
        mean_fpv.append(float(np.mean(tt[i])))
    dif = np.array(max_fpv) - np.array(min_fpv)

    # Replace zero difference with value 1 for devision.
    np.place(dif, dif == 0., [1.])

    norm = defaultdict(list)
    for i in train:
        norm['train'].append(np.array((i - np.array(mean_fpv)) / dif))
    if test is not None:
        for i in test:
            norm['test'].append(np.array((i - np.array(mean_fpv)) / dif))
    norm['mean'] = np.array(mean_fpv)
    norm['dif'] = dif

    return norm
    
def get_combined_feature_labels(atoms, labels):
    """ Function to sequentially combine fingerprint vectors and return them
        for a list of atoms objects.
    """
    # Check that there are at least two fingerprint descriptors to combine.
    msg = "This functions combines various fingerprint"
    msg += " vectors, there must be at least two to combine"
    assert len(labels) >= 2, msg
    fpv_list = labels[::-1]
    L_F = []
    for j in range(len(fpv_list)):
        fpv = (fpv_list[j](atoms))
        for i in range(len(fpv)):
            fpl = (str(fpv_list[j])).split(' of')[0].replace('<bound method ','')+'_'+str(i)
            L_F.append(fpl)
    return np.array(L_F)
