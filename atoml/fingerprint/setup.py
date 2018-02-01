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


def return_fpv(candidates, fpv_names):
    """Sequentially combine fingerprint vectors. Padding handled automatically.

    Parameters
    ----------
    candidates : list or dict
        Atoms objects to construct fingerprints for.
    fpv_name : list of / single fpv class(es)
        List of fingerprinting classes.

    Returns
    -------
    fingerprint_vector : ndarray
      Fingerprint array (n, m) where n is the number of candidates and m is the
      summed number of features from all fingerprint classes supplied.
    """
    if not isinstance(candidates, (list, defaultdict)):
        raise TypeError("return_fpv requires a list or dict of atoms")

    if not isinstance(fpv_names, list):
        fpv_names = [fpv_names]
    fpvn = len(fpv_names)

    maxatoms = np.argmax([len(atoms) for atoms in candidates])
    maxcomp = np.argmax(
        [len(set(atoms.get_chemical_symbols())) for atoms in candidates])

    # PATCH: Ideally fp length would be called from a property.
    fps = np.zeros(fpvn, dtype=int)
    for i, fp in enumerate(fpv_names):
        fps[i] = max(len(fp(candidates[maxatoms])),
                     len(fp(candidates[maxcomp])))

    fingerprint_vector = np.zeros((len(candidates), sum(fps)))
    for i, atoms in enumerate(candidates):
        fingerprint_vector[i] = _get_fpv(atoms, fpv_names, fps)

    return fingerprint_vector


def _get_fpv(atoms, fpv_names, fps):
    """Get the fingerprint vector as an array.

    Parameters
    ----------
    atoms : object
        A single atoms object.
    fpv_name : list of / single fpv class(es)
        List of fingerprinting classes.
    fps : list
        List of expected feature vector lengths.

    Returns
    -------
    fingerprint_vector : list
        A feature vector.
    """
    if len(fpv_names) == 1:
        fp = fpv_names[0](atoms=atoms)
        fingerprint_vector = np.zeros((fps[0]))
        fingerprint_vector[:len(fp)] = fp

    else:
        fingerprint_vector = _concatenate_fpv(atoms, fpv_names, fps)

    return fingerprint_vector


def _concatenate_fpv(atoms, fpv_names, fps):
    """Join multiple fingerprint vectors.

    Parameters
    ----------
    atoms : object
        A single atoms object.
    fpv_name : list of / single fpv class(es)
        List of fingerprinting classes.
    fps : list
        List of expected feature vector lengths.

    Returns
    -------
    fingerprint_vector : list
        A feature vector.
    """
    fingerprint_vector = np.zeros((sum(fps)))
    st = 0
    for i, name in enumerate(fpv_names):
        fi = sum(fps[:i+1])
        fp = name(atoms=atoms)
        fingerprint_vector[st:fi] = fp
        st = fi

    return fingerprint_vector
