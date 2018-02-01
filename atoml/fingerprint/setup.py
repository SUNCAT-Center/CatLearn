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


def return_fpv(candidates, fpv_names, use_prior=True):
    """ Sequentially combine fingerprint vectors. Padding is
    handled automatically.

   Parameters:
      candidates: list or dict of atoms-object(s)
        Atoms objects to construct fingerprints for.
      fpv_name: list of / single fpv class(es)
        List of fingerprinting classes
      use_prior: bool
        (requires documentation)

    Returns: ndarray (n, m)
      Fingerprint array where n is the number of candidates
      and m is the number of atoms times the length of fingerprint
      classes supplied.
    """

    if not isinstance(candidates, (list, defaultdict)):
        raise TypeError("return_fpv requires a list or dict of atoms")

    if not isinstance(fpv_names, list):
        fpv_names = [fpv_names]
    fpvn = len(fpv_names)

    # PATCH: Ideally fp_length would be called from a property.
    fps = np.zeros(fpvn, dtype=int)
    for i, fp in enumerate(fpv_names):
        fp_length = len(_get_fpv(candidates[0], [fp], use_prior))
        fps[i] = int(fp_length / len(candidates[0]))

    maxatoms = max([len(atoms) for atoms in candidates])
    maxm = sum(maxatoms * fps)

    fingerprint_vector = np.zeros((len(candidates), maxm))
    for i, atoms in enumerate(candidates):
        m = len(atoms) * fpvn
        fingerprint_vector[i, :m] = _get_fpv(atoms, fpv_names, use_prior)

    return fingerprint_vector


def _get_fpv(atoms, fpv_names, use_prior):
    """Get the fingerprint vector as an array.

    If a fingerprint vector is saved in info['data']['fpv'] it is returned
    otherwise saved in the data dictionary.
    """

    if len(fpv_names) == 1:
        if not use_prior:
            return fpv_names[0](atoms=atoms)

        if 'data' not in atoms.info:
            atoms.info['data'] = {'fpv': fpv_names[0](atoms=atoms)}
        elif 'fpv' not in atoms.info['data']:
            atoms.info['data']['fpv'] = fpv_names[0](atoms=atoms)

        return atoms.info['data']['fpv']

    if not use_prior:
        return _concatenate_fpv(atoms, fpv_names)

    if 'data' not in atoms.info:
        atoms.info['data'] = {'fpv': _concatenate_fpv(atoms, fpv_names)}
    elif 'fpv' not in atoms.info['data']:
        atoms.info['data']['fpv'] = _concatenate_fpv(atoms, fpv_names)

    return atoms.info['data']['fpv']


def _concatenate_fpv(atoms, fpv_name):
    """Join multiple fingerprint vectors."""

    fpv = fpv_name[0](atoms=atoms)
    for i in fpv_name[1:]:
        fpv = np.concatenate((i(atoms=atoms), fpv))

    return fpv
