# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:04:38 2017

This function constructs a dictionary with abinitio_energies.

Input:
    fname (str) path/filename of ase.db file
    selection (list) ase.db selection
"""

import numpy as np
import ase.db
from ase.atoms import string2symbols
from ase.data import covalent_radii
from ase.geometry import get_layers
# from ase.data import chemical_symbols


addsyms = ['H', 'C', 'O', 'N']


def slab_index(atoms):
    """ Return the index of all atoms that are not in atoms.info['ads_atoms']

    Parameters
    ----------
    atoms : ase atoms object
        atoms.info must be a dictionary containing the key 'ads_atoms'.
    """
    slab_atoms = [a.index for a in atoms if a.index not in
                  atoms.info['ads_atoms']]
    return slab_atoms


def ads_index(atoms):
    """ Returns the indexes of atoms that are in the global list addsyms.

    Parameters
    ----------
    atoms : ase atoms object.
    """
    add_atoms = [a.index for a in atoms if a.symbol in addsyms]

    return add_atoms


def last_index2ads(atoms, formula):
    """ Returns the indexes of the last n atoms in the atoms object, where n is
    the length of the composition of the species field.

    Parameters
    ----------
    atoms : ase atoms object.
        atoms.info must be a dictionary containing the key 'key_value_pairs',
        which is expected to contain standard adsorbate structure
        key value pairs. 'species' should be the chemical formula of an
        adsorbate.

    Parameters
    ----------
    atoms : ase atoms object.
    """
    n_ads = len(string2symbols(formula))
    natoms = len(atoms)
    add_atoms = list(range(natoms-n_ads, natoms))
    return add_atoms


def detect_ads(atoms, formula):
    """ Returns the indexes of the last n atoms in the atoms object, where n is
    the length of the composition of the species field.

    Parameters
    ----------
    atoms : ase atoms object.
        atoms.info must be a dictionary containing the key 'key_value_pairs',
        which is expected to contain standard adsorbate structure
        key value pairs. 'species' should be the chemical formula of an
        adsorbate.

    Parameters
    ----------
    atoms : ase atoms object.
    """
    string2symbols(formula)
    n_ads = len()
    natoms = len(atoms)
    lz, li = get_layers(atoms, (0, 0, 1), tolerance=0.3)
    layers = int(atoms.info['key_value_pairs']['layers'])
    add_atoms = [a.index for a in atoms if li[a.index] > layers-1]
    add_atoms = list(range(natoms-n_ads, natoms))
    return add_atoms


def layers_info(atoms):
    """Returns two lists, the first containing indices of subsurface atoms and
    the second containing indices of atoms in the two outermost layers.

    Parameters
    ----------
    atoms : ase atoms object.
    """
    il, z = get_layers(atoms, (0, 0, 1),
                       tolerance=covalent_radii[atoms.info['Z_surf1']])
    layers = atoms.info['key_value_pairs']['layers']
    if len(z) < layers or len(z) > layers * 2:
        top_atoms = atoms.info['surf_atoms']
        bulk_atoms = atoms.info['surf_atoms']
    else:
        bulk_atoms = [a.index for a in atoms
                      if il[a.index] < layers-2]
        top_atoms = [a.index for a in atoms
                     if il[a.index] > layers-3 and
                     a.index not in atoms.info['ads_atoms']]
    assert len(bulk_atoms) > 0 and len(top_atoms) > 0
    return bulk_atoms, top_atoms


def info2primary_index(atoms, rtol=1.3):
    """ Returns lists identifying the nearest neighbors of the adsorbate atoms.

    Parameters
    ----------
    atoms : ase atoms object.
        atoms.info must be a dictionary containing the keys 'ads_atoms' and
        'surf_atoms'.
    """
    liste = []
    surf_atoms = atoms.info['surf_atoms']
    add_atoms = atoms.info['ads_atoms']
    for m in surf_atoms:
        dM = covalent_radii[atoms.numbers[m]]
        for a in add_atoms:
            dA = covalent_radii[atoms.numbers[a]]
            # Covalent radii are subtracted in distance comparison.
            d = atoms.get_distance(m, a, mic=True, vector=False)-dM-dA
            liste.append([a, m, d])
    L = np.array(liste)
    i = np.argmin(L[:, 2])
    # dmin = L[i, 2]
    i_add1 = L[i, 0]
    i_surf1 = L[i, 1]
    Z_add1 = atoms.numbers[int(i_add1)]
    Z_surf1 = atoms.numbers[int(i_surf1)]
    dadd = covalent_radii[Z_add1]
    # i_surfnn = [a.index for a in atoms if a.symbol not in addsyms and
    #            atoms.get_distance(int(i_add1),
    #                               int(a.index), mic=True) < dmin * rtol]
    i_surfnn = [a.index for a in atoms if a.symbol not in addsyms and
                atoms.get_distance(int(i_add1), int(a.index), mic=True) <
                (covalent_radii[a.number]+dadd) * rtol]
    return i_add1, i_surf1, Z_add1, Z_surf1, i_surfnn


def db2surf(fname, selection=[]):
    csurf = ase.db.connect(fname)
    ssurf = csurf.select(selection)
    abinitio_energies = {}
    dbids = {}
    # Get slabs and speciess from .db.
    for d in ssurf:
        cat = str(d.name) + '_' + str(d.phase)
        facet = str(d.facet)
        lattice = str(d.surf_lattice)
        abinitio_energy = float(d.epot)
        ads = str(d.ads)
        species = str(d.species)
        if '-' in species:
            continue
        if ads == 'clean':
            ser = ''
            site = 'slab'
        else:
            ser = species
            site = str(d.site)
        size = str(d.supercell) + 'x' + str(d.layers)
        n = str(d.n)
        site_name = lattice + '_' + facet + '_' + size + '_' + site
        key = n + '_' + ser + '_' + cat + '_' + site_name
        if key not in abinitio_energies:
            abinitio_energies[key] = abinitio_energy
            dbids[key] = int(d.id)
        elif abinitio_energies[key] > abinitio_energy:
            abinitio_energies[key] = abinitio_energy
            dbids[key] = int(d.id)
    return abinitio_energies, dbids


# The variable fname must be path/filename of db containing molecules.
def db2mol(fname, selection=[]):
    cmol = ase.db.connect(fname)
    smol = cmol.select(selection)
    # mol_dict = {}
    abinitio_energies = {}
    dbids = {}
    # Get molecules from mol.db.
    for d in smol:
        abinitio_energy = float(d.epot)
        species_name = str(d.formula)
        if species_name+'_gas' not in abinitio_energies:
            abinitio_energies[species_name+'_gas'] = abinitio_energy
            dbids[species_name+'_gas'] = int(d.id)
        elif abinitio_energies[species_name+'_gas'] > abinitio_energy:
            abinitio_energies[species_name+'_gas'] = abinitio_energy
            dbids[species_name+'_gas'] = int(d.id)
    return abinitio_energies, dbids


def mol2ref(abinitio_energies):
    mol_dict = {}
    mol_dict['H'] = 0.5*abinitio_energies['H2_gas']
    mol_dict['O'] = abinitio_energies['H2O_gas'] - 2*mol_dict['H']
    mol_dict['C'] = abinitio_energies['CH4_gas'] - 4*mol_dict['H']
    # mol_dict['C'] = abinitio_energies['CO_gas'] - mol_dict['O']
    return mol_dict


# Adapted from CATMAP wiki.
def get_refs(energy_dict, mol_dict):
    ref_dict = mol_dict
    for key in energy_dict.keys():
        if 'slab' in key:
            n, ser, cat, pha, lattice, fac, size, site = key.split('_')
            Eref = energy_dict[key]
            name = n + '_' + ser + '_' + cat + '_' + pha + '_' + lattice + \
                '_' + fac + '_' + size + '_slab'
            ref_dict[name] = Eref
    return ref_dict


def get_formation_energies(energy_dict, ref_dict):  # adapted from CATMAP wiki
    formation_energies = {}
    for key in energy_dict.keys():
        E0 = 0
        if 'gas' in key:
            ser, site_name = key.split('_')
        else:
            n, ser, cat, pha, lattice, fac, size, site = key.split('_')
            site_name = '0__' + cat + '_' + pha+'_' + lattice + '_' + fac + \
                '_' + size + '_slab'
            if site_name in ref_dict:
                E0 -= ref_dict[site_name]
            else:
                print('no slab reference '+site_name)
                continue
        if 'slab' not in key:
            try:
                composition = string2symbols(ser)
            except ValueError:
                print(ser, cat, pha, lattice, fac, site)
                raise
            E0 += energy_dict[key]
            for atom in composition:
                E0 -= ref_dict[atom]
            formation_energies[key] = round(E0, 4)
    return formation_energies


def db2surf_info(fname, id_dict, formation_energies=None):
    """ Returns a list of atoms objects including only the most stable
        species state for each key in the dict self.dbids.

        Also attaches the required atoms.info to species states.
    """
    c = ase.db.connect(fname)
    traj = []
    for key in id_dict:
        dbid = id_dict[key]
        d = c.get(dbid)
        atoms = c.get_atoms(dbid)
        atoms.info['key_value_pairs'] = d.key_value_pairs
        atoms.info['dbid'] = dbid
        atoms.info['ctime'] = float(d.ctime)
        species = atoms.info['key_value_pairs']['species']
        if species == '':
            print('Warning: No adsorbate.', fname, dbid)
            atoms.info['ads_atoms'] = []
            continue
        else:
            atoms.info['ads_atoms'] = ads_index(atoms)
            #  last_index2ads(atoms, species)
        atoms.info['surf_atoms'] = slab_index(atoms)
        i_add1, i_surf1, Z_add1, Z_surf1, i_surfnn = info2primary_index(atoms)
        atoms.info['i_add1'] = i_add1
        atoms.info['i_surf1'] = i_surf1
        atoms.info['Z_add1'] = Z_add1
        atoms.info['Z_surf1'] = Z_surf1
        atoms.info['i_surfnn'] = i_surfnn
        if formation_energies is not None:
            try:
                atoms.info['Ef'] = formation_energies[key]
            except KeyError:
                atoms.info['Ef'] = np.NaN
                print(key, 'does not have Ef')
        traj.append(atoms)
    return traj


def db2atoms_info(fname, selection=[]):
    """ Returns a list of atoms objects.
        Attaches the required atoms.info to species states.
    """
    c = ase.db.connect(fname)
    s = c.select(selection)
    traj = []
    for d in s:
        dbid = int(d.id)
        atoms = c.get_atoms(dbid)
        atoms.info['key_value_pairs'] = d.key_value_pairs
        atoms.info['ctime'] = float(d.ctime)
        atoms.info['dbid'] = int(d.id)
        # species = atoms.info['key_value_pairs']['species']
        atoms.info['ads_atoms'] = ads_index(atoms)
        # last_index2ads(atoms, species)
        atoms.info['surf_atoms'] = slab_index(atoms)
        i_add1, i_surf1, Z_add1, Z_surf1, i_surfnn = info2primary_index(atoms)
        atoms.info['i_add1'] = i_add1
        atoms.info['i_surf1'] = i_surf1
        atoms.info['Z_add1'] = Z_add1
        atoms.info['Z_surf1'] = Z_surf1
        atoms.info['i_surfnn'] = i_surfnn
        traj.append(atoms)
    return traj