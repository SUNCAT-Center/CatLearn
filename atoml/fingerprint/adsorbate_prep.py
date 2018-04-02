# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:04:38 2017

This function constructs a dictionary with abinitio_energies.

Input:
    fname (str) path/filename of ase.db file
    selection (list) ase.db selection
"""

import numpy as np
from ase.atoms import string2symbols
from ase.geometry import get_layers
from .periodic_table_data import get_radius
from atoml.fingerprint.base import BaseGenerator


addsyms = ['H', 'C', 'O', 'N', 'S']
gen = BaseGenerator()
dx = {}
for z in range(1, 93):
    dx.update({z: get_radius(z)})


def slab_index(atoms):
    """ Return the index of all atoms that are not in atoms.info['ads_atoms']

    Parameters
    ----------
    atoms : ase atoms object
        atoms.info must be a dictionary containing the key 'ads_atoms'.
    """
    chemi = [a.index for a in atoms if a.index not in atoms.info['ads_atoms']]
    return chemi


def sym2ads_index(atoms):
    """ Returns the indexes of atoms that are in the global list addsyms.

    Parameters
    ----------
    atoms : ase atoms object.
    """
    add_atoms = [a.index for a in atoms if a.symbol in addsyms]

    return add_atoms


def last2ads_index(atoms, formula):
    """ Returns the indexes of the last n atoms in the atoms object, where n is
    the length of the composition of the adsorbate species. This function will
    work on atoms objects, where the slab was set up first,
    and the adsorbate was added after.

    Parameters
    ----------
    atoms : ase atoms object.
        atoms.info must be a dictionary containing the key 'key_value_pairs',
        which is expected to contain CATMAP standard adsorbate structure
        key value pairs. See the ase db to catmap module in catmap.
        the key value pair 'species' must be the
        chemical formula of the adsorbate.
    """
    n_ads = len(string2symbols(formula))
    natoms = len(atoms)
    add_atoms = list(range(natoms - n_ads, natoms))
    composition = string2symbols(formula)
    for a in add_atoms:
        if atoms[a].symbol not in composition:
            raise AssertionError("last index adsorbate identification failed.")
    return add_atoms


def formula2ads_index(atoms, formula):
    """ Returns the indexes of atoms,
    which have symbols matching the chemical formula of the adsorbate. This
    function will not work for adsorbates containing the same elements as the
    slab.

    Parameters
    ----------
    atoms : ase atoms object.
        atoms.info must be a dictionary containing the key 'key_value_pairs',
        which is expected to contain CatMAP standard adsorbate structure
        key value pairs. See the ase db to catmap module in catmap.
        the key value pair 'species' must be the
        chemical formula of the adsorbate.
    formula: str
        chemical formula of the adsorbate.
    """
    try:
        composition = string2symbols(formula)
    except ValueError:
        print(formula)
        raise
    ads_atoms = [a.index for a in atoms if a.symbol in composition]
    if len(ads_atoms) != len(composition):
        raise AssertionError("ads atoms identification by formula failed.")
    return ads_atoms


def layers2ads_index(atoms, formula=None):
    """ Returns the indexes of atoms in layers exceeding the number of layers
    stored in the key value pair 'layers'.

    Parameters
    ----------
    atoms : ase atoms object.
        atoms.info must be a dictionary containing the key 'key_value_pairs',
        which is expected to contain CATMAP standard adsorbate structure
        key value pairs. See the ase db to catmap module in catmap.
        the key value pair 'species' must be the
        chemical formula of the adsorbate and 'layers' must be an integer.
    """
    composition = string2symbols(formula)
    n_ads = len(composition)
    natoms = len(atoms)
    radii = [get_radius(s) for s in composition]
    lz, li = get_layers(atoms, (0, 0, 1), tolerance=2 * min(radii))
    layers = int(atoms.info['key_value_pairs']['layers'])
    ads_atoms = [a.index for a in atoms if li[a.index] > layers - 1]
    ads_atoms = list(range(natoms - n_ads, natoms))
    if len(ads_atoms) != len(composition):
        raise AssertionError("ads atoms identification by layers failed.")
    return ads_atoms


def z2ads_index(atoms, formula):
    """ Returns the indexes of the n atoms with the highest position
    in the z direction,
    where n is the number of atoms in the chemical formula from the species
    key value pair.

    Parameters
    ----------
    atoms : ase atoms object.
        atoms.info must be a dictionary containing the key 'key_value_pairs',
        which is expected to contain CATMAP standard adsorbate structure
        key value pairs. See the ase db to catmap module in catmap.
        the key value pair 'species'.
    """
    composition = string2symbols(formula)
    n_ads = len(composition)
    z = atoms.get_positions()[:, 2]
    ads_atoms = np.argsort(z)[:n_ads]
    for a in ads_atoms:
        if atoms[a].symbol not in composition:
            raise AssertionError("adsorbate identification by z coord failed.")
    return ads_atoms


def layers_info(atoms):
    """Returns two lists, the first containing indices of subsurface atoms and
    the second containing indices of atoms in the two outermost layers.

    Parameters
    ----------
    atoms : ase atoms object.
    """
    radii = [get_radius(z) for z in atoms.numbers[atoms.info['site_atoms']]]
    radius = np.average(radii)
    il, z = get_layers(atoms, (0, 0, 1),
                       tolerance=radius)
    layers = atoms.info['key_value_pairs']['layers']
    if len(z) < layers or len(z) > layers * 2:
        top_atoms = atoms.info['slab_atoms']
        bulk_atoms = atoms.info['slab_atoms']
    else:
        bulk_atoms = [a.index for a in atoms
                      if il[a.index] < layers - 2]
        top_atoms = [a.index for a in atoms
                     if il[a.index] > layers - 3 and
                     a.index not in atoms.info['ads_atoms']]
    assert len(bulk_atoms) > 0 and len(top_atoms) > 0
    return bulk_atoms, top_atoms


def info2primary_index(atoms, rtol=1.3):
    """ Returns lists identifying the nearest neighbors of the adsorbate atoms.

    Parameters
    ----------
    atoms : ase atoms object.
        atoms.info must be a dictionary containing the keys 'ads_atoms' and
        'slab_atoms'.
    """
    slab_atoms = atoms.info['slab_atoms']
    ads_atoms = atoms.info['ads_atoms']
    gen.make_neighborlist(atoms, dx=dx)
    nl = gen.get_neighborlist(atoms)
    chemi = []
    site = []
    ligand = []
    for a_a in ads_atoms:
        for a_s in slab_atoms:
            if int(a_a) in nl[a_s]:
                site.append(a_s)
                chemi.append(a_a)
    for j in site:
        for a_s in slab_atoms:
            if j in nl[a_s]:
                ligand.append(a_s)
    return chemi, site, ligand


def catalysis_hub_to_info(images):
    raise NotImplementedError("Coming soon.")


def autogen_adsorbate_info(images):
    """Returns an atoms object with essential neighbor information attached.

    Todo: checks for attached graph representations or neighborlists.

    Parameters
    ----------
        traj : list
            List of ASE atoms objects."""
    traj = []
    for i, atoms in enumerate(images):
        try:
            species = atoms.info['key_value_pairs']['species']
            try:
                atoms.info['ads_atoms'] = formula2ads_index(atoms, species)
            except AssertionError:
                atoms.info['ads_atoms'] = last2ads_index(atoms, species)
        except KeyError:
            pass
        atoms.info['slab_atoms'] = slab_index(atoms)
        chemi, site, ligand = info2primary_index(atoms)
        atoms.info['chemisorbed_atoms'] = chemi
        atoms.info['site_atoms'] = site
        atoms.info['ligand_atoms'] = ligand
        traj.append(atoms)
    return traj
