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
from ase.geometry import get_layers
from .periodic_table_data import get_mendeleev_params

addsyms = ['H', 'C', 'O', 'N', 'S']


def get_radius(z):
    p = get_mendeleev_params(z, ['atomic_radius'])
    if p[-1] is not None:
        r = p[-1]
    elif p[-4] is not None:
        r = p[-4]
    return float(r) / 100.


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
    add_atoms = list(range(natoms-n_ads, natoms))
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
    ads_atoms = [a.index for a in atoms if li[a.index] > layers-1]
    ads_atoms = list(range(natoms-n_ads, natoms))
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
    radius = get_radius(atoms.info['Z_surf1'])
    il, z = get_layers(atoms, (0, 0, 1),
                       tolerance=radius)
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
        dM = get_radius(atoms.numbers[m])
        for a in add_atoms:
            dA = get_radius(atoms.numbers[a])
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
    dadd = get_radius(Z_add1)
    # i_surfnn = [a.index for a in atoms if a.symbol not in addsyms and
    #            atoms.get_distance(int(i_add1),
    #                               int(a.index), mic=True) < dmin * rtol]
    i_surfnn = [a.index for a in atoms if a.symbol not in addsyms and
                atoms.get_distance(int(i_add1), int(a.index), mic=True) <
                (get_radius(a.number) + dadd) * rtol]
    return i_add1, i_surf1, Z_add1, Z_surf1, i_surfnn


def db2surf(fname, selection=[]):
    csurf = ase.db.connect(fname)
    ssurf = csurf.select(selection)
    abinitio_energies = {}
    dbids = {}
    # Get slabs and speciess from .db.
    for d in ssurf:
        abinitio_energy = float(d.epot)
        ads = str(d.ads)
        species = str(d.species)
        if '-' in species:
            continue
        if ads == 'clean' or species == '':
            ser = ''
            site = 'slab'
            n = '0'
        else:
            ser = species
            site = str(d.site)
            if 'n' in d:
                n = str(d.n)
            else:
                n = '1'
        if 'phase' in d:
            phase = str(d.phase)
        elif 'crystal' in d:
            phase = str(d.crystal)
        else:
            phase = ''
        if 'facet' in d:
            facet = str(d.facet)
        else:
            facet = 'facet'
        if 'surf_lattice' in d:
            lattice = str(d.surf_lattice)
        else:
            lattice = 'lattice'
        if 'supercell' in d:
            cell = str(d.supercell)
        else:
            cell = 'XxY'
        if 'layers' in d:
            cell += 'x' + str(d.layers)
        cat = str(d.name) + '_' + phase
        site_name = lattice + '_' + facet + '_' + cell + '_' + site
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
    mol_dict['S'] = abinitio_energies['O2S_gas'] - 2*mol_dict['O']
    mol_dict['N'] = abinitio_energies['H3N_gas'] - 3*mol_dict['H']
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
    missing_slabs = []
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
                missing_slabs.append(site_name)
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
    print('missing slabs:', missing_slabs)
    return formation_energies


def db2surf_info(fname, id_dict, formation_energies=None):
    """ Returns a list of atoms objects including only the most stable
        species state for each key in the dict self.dbids.

        Also attaches the required atoms.info to species states.
    """
    c = ase.db.connect(fname)
    traj = []
    failed = []
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
            atoms.info['ads_atoms'] = formula2ads_index(atoms, species)
        atoms.info['surf_atoms'] = slab_index(atoms)
        i_add1, i_surf1, Z_add1, Z_surf1, i_surfnn = info2primary_index(atoms)
        atoms.info['i_add1'] = i_add1
        atoms.info['i_surf1'] = i_surf1
        atoms.info['Z_add1'] = Z_add1
        atoms.info['Z_surf1'] = Z_surf1
        atoms.info['i_surfnn'] = i_surfnn
        if formation_energies is not None:
            if 'Ef' in atoms.info:
                atoms.info['Ef'] = formation_energies[key]
            else:
                atoms.info['Ef'] = np.NaN
                failed.append(str(dbid))
        traj.append(atoms)
    print('db ids', ','.join(failed), ' are missing slab reference.')
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
        species = atoms.info['key_value_pairs']['species']
        atoms.info['ads_atoms'] = formula2ads_index(atoms, species)
        atoms.info['surf_atoms'] = slab_index(atoms)
        i_add1, i_surf1, Z_add1, Z_surf1, i_surfnn = info2primary_index(atoms)
        atoms.info['i_add1'] = i_add1
        atoms.info['i_surf1'] = i_surf1
        atoms.info['Z_add1'] = Z_add1
        atoms.info['Z_surf1'] = Z_surf1
        atoms.info['i_surfnn'] = i_surfnn
        traj.append(atoms)
    return traj


def db2unlabeled_atoms(fname, selection=[]):
    """ Returns a list of atoms objects.
    """
    c = ase.db.connect(fname)
    s = c.select(selection)
    traj = []
    for d in s:
        dbid = int(d.id)
        atoms = c.get_atoms(dbid)
        traj.append(atoms)
    return traj


def attach_adsorbate_info(images):
    """Returns an atoms object with essential neighbor information attached.

    Todo: checks for attached graph representations or neighborlists.

    Parameters
    ----------
        traj : list
            List of ASE atoms objects."""
    traj = []
    for i, atoms in enumerate(images):
        if 'key_value_pairs' not in atoms.info:
            atoms.info['key_value_pairs'] = {}
            if 'species' in atoms.info:
                atoms.info['key_value_pairs']['species'] = \
                    atoms.info['species']
            if 'layers' in atoms.info:
                atoms.info['key_value_pairs']['layers'] = atoms.info['layers']
        species = atoms.info['key_value_pairs']['species']
        try:
            atoms.info['ads_atoms'] = formula2ads_index(atoms, species)
        except AssertionError:
            atoms.info['ads_atoms'] = last2ads_index(atoms, species)
        atoms.info['surf_atoms'] = slab_index(atoms)
        i_add1, i_surf1, Z_add1, Z_surf1, i_surfnn = info2primary_index(atoms)
        atoms.info['i_add1'] = i_add1
        atoms.info['i_surf1'] = i_surf1
        atoms.info['Z_add1'] = Z_add1
        atoms.info['Z_surf1'] = Z_surf1
        atoms.info['i_surfnn'] = i_surfnn
        traj.append(atoms)
    return traj
