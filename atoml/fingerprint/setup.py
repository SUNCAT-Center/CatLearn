"""Functions to setup fingerprint vectors."""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
from collections import defaultdict

import ase.db
from .particle_fingerprint import ParticleFingerprintGenerator
from .adsorbate_fingerprint import AdsorbateFingerprintGenerator


def db_sel2fp(calctype, fname, selection, moldb=None, bulkdb=None,
              slabref=None):
    """Return array of fingerprints from ase.db files and selection.

    Parameters
    ----------
    calctype : str
    fname : str
    selection : list
    moldb : str
    bulkdb : str
    DFT_parameters : dict
    """
    keys = {}
    c = ase.db.connect(fname)
    s = c.select(selection)
    for d in s:
        keys.update(d.key_value_pairs)
    k = list(keys)
    print(k)
    fpv = []
    if calctype == 'adsorption':
        assert moldb is not None
        if 'enrgy' in k:
            if slabref is None:
                slabs = fname
            fpv_gen = AdsorbateFingerprintGenerator(moldb=moldb, bulkdb=bulkdb,
                                                    slabs=slabs)
            cand = fpv_gen.db2adds_info(fname=fname, selection=selection)
            fpv += [fpv_gen.get_Ef]
        else:
            fpv_gen = AdsorbateFingerprintGenerator(moldb=moldb, bulkdb=bulkdb)
            cand = fpv_gen.db2adds_info(fname=fname, selection=selection)
        fpv += [fpv_gen.Z_add,
                fpv_gen.primary_addatom,
                fpv_gen.primary_adds_nn,
                fpv_gen.adds_sum,
                fpv_gen.primary_surfatom]
        if bulkdb is not None:
            fpv += [fpv_gen.primary_surf_nn,
                    fpv_gen.elemental_dft_properties]
        cfpv = return_fpv(cand, fpv)
    elif calctype == 'nanoparticle':
        fpv_gen = ParticleFingerprintGenerator()
        fpv += [fpv_gen.atom_numbers,
                fpv_gen.bond_count_fpv,
                fpv_gen.connections_fpv,
                fpv_gen.distribution_fpv,
                fpv_gen.nearestneighbour_fpv,
                fpv_gen.rdf_fpv]
        cfpv = return_fpv(cand, fpv)
    fpv_labels = get_combined_descriptors(fpv)
    return cfpv, fpv_labels


def get_combined_descriptors(fpv_list):
    """Sequentially combine feature label vectors.

    Parameters
    ----------
    fpv_list : list
        Functions that return fingerprints.
    """
    # Check that there are at least two fingerprint descriptors to combine.
    msg = "This functions combines various fingerprint"
    msg += " vectors, there must be at least two to combine"
    assert len(fpv_list) >= 2, msg
    labels = fpv_list[::-1]
    L_F = []
    for j in range(len(labels)):
        L_F.append(labels[j]())
    return np.hstack(L_F)


def get_keyvaluepair(c=[], fpv_name='None'):
    """Get a list of the key_value_pairs target names/values."""
    if len(c) == 0:
        return ['kvp_' + fpv_name]
    else:
        out = []
        for atoms in c:
            field_value = float(atoms['key_value_pairs'][fpv_name])
            out.append(field_value)
        return out


def return_fpv(candidates, fpv_name, use_prior=True, max_length=None):
    """Sequentially combine fingerprint vectors."""
    # Put fpv_name in a list, if it is not already.
    if not isinstance(fpv_name, list):
        fpv_name = [fpv_name]

    # Check to see if we are dealing with a list of candidates or a single
    # atoms object.
    if type(candidates) is defaultdict or type(candidates) is list:
        list_fp = []
        for c in candidates:
            fpv = list(_get_fpv(c, fpv_name, use_prior))
            if max_length is not None:
                fpv = _pad_zero(fpv, max_length)
            list_fp.append(fpv)
        return np.asarray(list_fp)
    # Do the same but for a single atoms object.
    else:
        c = candidates
        fpv = list(_get_fpv(c, fpv_name, use_prior))
        if max_length is not None:
            fpv = _pad_zero(fpv, max_length)
        return np.asarray([fpv])


def _get_fpv(c, fpv_name, use_prior):
    """Get the fingerprint vector as an array.

    If a fingerprint vector is saved in info['data']['fpv'] it is returned
    otherwise saved in the data dictionary.
    """
    if len(fpv_name) == 1:
        if not use_prior:
            return fpv_name[0](atoms=c)
        if 'data' not in c.info:
            c.info['data'] = {'fpv': fpv_name[0](atoms=c)}
        elif 'fpv' not in c.info['data']:
            c.info['data']['fpv'] = fpv_name[0](atoms=c)
        return c.info['data']['fpv']
    if not use_prior:
        return _concatenate_fpv(c, fpv_name)
    if 'data' not in c.info:
        c.info['data'] = {'fpv': _concatenate_fpv(c, fpv_name)}
    elif 'fpv' not in c.info['data']:
        c.info['data']['fpv'] = _concatenate_fpv(c, fpv_name)
    return c.info['data']['fpv']


def _concatenate_fpv(c, fpv_name):
    """Join multiple fingerprint vectors."""
    fpv = fpv_name[0](atoms=c)
    for i in fpv_name[1:]:
        fpv = np.concatenate((i(atoms=c), fpv))
    return fpv


def _pad_zero(fpv, max_length):
    """Function to pad features with zeros."""
    if len(fpv) < max_length:
        p = [0.] * (max_length - len(fpv))
        return fpv + p
