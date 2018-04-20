# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:04:38 2017

This function constructs a dictionary with abinitio_energies.

Input:
    fname (str) path/filename of ase.db file
    selection (list) ase.db selection
"""
import warnings
import numpy as np
from tqdm import tqdm
from ase.atoms import string2symbols
from ase.geometry import get_layers
from atoml.api.ase_atoms_api import extend_atoms_class
from atoml.utilities.neighborlist import ase_neighborlist
from .periodic_table_data import get_radius, default_atoml_radius


ads_syms = ['H', 'C', 'O', 'N', 'S', 'F', 'Cl']


def catalysis_hub_to_info(images):
    raise NotImplementedError("Coming soon.")


def autogen_info(images):
    """ Returns a list of atoms objects with atomic group information
    attached to info.
    This information is  needed by some functions in the
    AdsorbateFingerprintGenerator.

    Parameters
    ----------
    images : list
        list of atoms objects representing adsorbates on slabs.
        No further information is required in atoms.info.
    """
    traj = []
    for atoms in tqdm(images):
        if not hasattr(atoms, 'atoml') or 'neighborlist' not in atoms.atoml:
            extend_atoms_class(atoms)
            radii = [default_atoml_radius(z) for z in atoms.numbers]
            nl = ase_neighborlist(atoms, cutoffs=radii)
            atoms.set_neighborlist(nl)
        if 'ads_atoms' not in atoms.info:
            ads_atoms = detect_adsorbate(atoms)
            if ads_atoms is None:
                continue
            else:
                atoms.info['ads_atoms'] = ads_atoms
        if 'slab_atoms' not in atoms.info:
            atoms.info['slab_atoms'] = slab_index(atoms)
        if ('chemisorbed_atoms' not in atoms.info or
            'site_atoms' not in atoms.info or
           'ligand_atoms' not in atoms.info):
            chemi, site, ligand = info2primary_index(atoms)
            atoms.info['chemisorbed_atoms'] = chemi
            atoms.info['site_atoms'] = site
            atoms.info['ligand_atoms'] = ligand
        if ('key_value_pairs' not in atoms.info or
            'term' not in atoms.info['key_value_pairs'] or
           'bulk' not in atoms.info['key_value_pairs']):
            bulk, subsurf, term = detect_termination(atoms)
            atoms.info['bulk_atoms'] = bulk
            atoms.info['termination_atoms'] = term
            atoms.info['subsurf_atoms'] = subsurf
        traj.append(atoms)
    return traj


def attach_nl(images):
    """ Return a list of atoms objects with attached neighborlists."""
    traj = []
    for atoms in tqdm(images):
        if not hasattr(atoms, 'atoml') or 'neighborlist' not in atoms.atoml:
            extend_atoms_class(atoms)
            radii = [default_atoml_radius(z) for z in atoms.numbers]
            nl = ase_neighborlist(atoms, cutoffs=radii)
            atoms.set_neighborlist(nl)
    return traj


def detect_adsorbate(atoms):
    """ Returns a list of indices of atoms belonging to an adsorbate.

    Parameters
    ----------
    atoms : ase atoms object
    """
    try:
        species = atoms.info['key_value_pairs']['species']
    except KeyError:
        warnings.warn("'species' key missing.")
        return sym2ads_index(atoms)
    if species == '':
        return []
    elif '-' in species or '+' in species or ',' in species:
        msg = "Co-adsorption not yet supported."
        if 'id' in atoms.info:
            msg += ' id: ' + str(atoms.info['id'])
        warnings.warn(msg)
        return None
        # raise NotImplementedError(msg)
    try:
        return formula2ads_index(atoms, species)
    except AssertionError:
        return last2ads_index(atoms, species)


def termination_info(images):
    """ Returns a list of atoms objects with attached information about
    the slab termination, the slab second outermost layer and the bulk slab
    compositions.

    Parameters
    ----------
    images : list
        list of atoms objects representing adsorbates on slabs.
        The atoms objects must have the following keys in atoms.info:
            - 'ads_atoms' : list
                indices of atoms belonging to the adsorbate
            - 'slab_atoms' : list
                indices of atoms belonging to the slab
    """
    traj = []
    for atoms in tqdm(images):
        bulk, subsurf, term = detect_termination(atoms)
        atoms.info['bulk_atoms'] = bulk
        atoms.info['termination_atoms'] = term
        atoms.info['subsurf_atoms'] = subsurf
        traj.append(atoms)
    return


def slab_index(atoms):
    """ Returns a list of indices of atoms belonging to the slab.
    These are defined as atoms that are not belonging to the adsorbate.

    Parameters
    ----------
    atoms : ase atoms object
        The atoms object must have the key 'ads_atoms' in atoms.info:
            - 'ads_atoms' : list
                indices of atoms belonging to the adsorbate
    """
    chemi = [a.index for a in atoms if a.index not in atoms.info['ads_atoms']]
    return chemi


def sym2ads_index(atoms):
    """ Returns the indexes of atoms that are in the global list addsyms.

    Parameters
    ----------
    atoms : ase atoms object.
    """
    ads_atoms = [a.index for a in atoms if a.symbol in ads_syms]

    return ads_atoms


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
    ads_atoms = list(range(natoms - n_ads, natoms))
    composition = string2symbols(formula)
    for a in ads_atoms:
        if atoms[a].symbol not in composition:
            raise AssertionError("last index adsorbate identification failed.")
    return ads_atoms


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


def layers2ads_index(atoms, formula):
    """ Returns the indexes of atoms in layers exceeding the number of layers
    stored in the key value pair 'layers'.

    Parameters
    ----------
    atoms : ase atoms object.
        atoms.info must be a dictionary containing the key 'key_value_pairs',
        which is expected to contain CatMAP standard adsorbate structure
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
        which is expected to contain CatMAP standard adsorbate structure
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


def info2primary_index(atoms):
    """ Returns lists identifying the nearest neighbors of the adsorbate atoms.

    Parameters
    ----------
    atoms : ase atoms object.
        The atoms object must have the following keys in atoms.info:
            - 'ads_atoms' : list
                indices of atoms belonging to the adsorbate
            - 'slab_atoms' : list
                indices of atoms belonging to the slab
    """
    slab_atoms = atoms.info['slab_atoms']
    ads_atoms = atoms.info['ads_atoms']
    nl = atoms.get_neighborlist()
    chemi = []
    site = []
    ligand = []
    for a_a in ads_atoms:
        for a_s in slab_atoms:
            if int(a_s) in nl[a_a]:
                site.append(a_s)
                chemi.append(a_a)
    for j in site:
        for a_s in slab_atoms:
            if j in nl[a_s]:
                ligand.append(a_s)
    if len(chemi) is 0 or len(site) is 0:
        print(chemi, site, ligand)
        msg = 'No adsorption site detected.'
        if 'id' in atoms.info:
            msg += ' dbid: ' + str(atoms.info['id'])
        warnings.warn(msg)
    elif len(ligand) < 2:
        msg = 'Site has less than 2 nearest neighbors.'
        if 'id' in atoms.info:
            msg += ' dbid: ' + str(atoms.info['id'])
        warnings.warn(msg)
    return chemi, site, ligand


def detect_termination(atoms):
    """ Returns three lists, the first containing indices of bulk atoms and
    the second containing indices of atoms in the second outermost layer, and
    the last denotes atoms in the outermost layer or termination or the slab.

    Parameters
    ----------
    atoms : ase atoms object.
        The atoms object must have the following keys in atoms.info:
            - 'ads_atoms' : list
                indices of atoms belonging to the adsorbate
            - 'slab_atoms' : list
                indices of atoms belonging to the slab
    """
    max_coord = 0
    try:
        nl = atoms.get_neighborlist()
        # List coordination numbers of slab atoms.
        coord = np.empty(len(atoms.info['slab_atoms']), dtype=int)
        for i, a_s in enumerate(atoms.info['slab_atoms']):
            coord[i] = len(nl[a_s])
            # Do not count adsorbate atoms.
            for a_a in atoms.info['ads_atoms']:
                if a_a in nl[a_s]:
                    coord[i] -= 1
        bulk = []
        term = []
        max_coord = max(coord)
        for j, c in enumerate(coord):
            a_s = atoms.info['slab_atoms'][j]
            if (c < max_coord and a_s not in
               atoms.constraints[0].get_indices()):
                term.append(a_s)
            else:
                bulk.append(a_s)
        subsurf = []
        for a_b in bulk:
            for a_t in term:
                if a_b in nl[a_t]:
                    subsurf.append(a_b)
        subsurf = list(np.unique(subsurf))
        try:
            sx, sy = atoms.info['key_value_pairs']['supercell'].split('x')
            sx = int(''.join(i for i in sx if i.isdigit()))
            sy = int(''.join(i for i in sy if i.isdigit()))
            if int(sx) * int(sy) != len(term):
                msg = str(len(term)) + ' termination atoms identified.' + \
                    'size = ' + atoms.info['key_value_pairs']['supercell'] + \
                    ' id: ' + str(atoms.info['id'])
                warnings.warn(msg)
        except KeyError:
            pass
        if len(bulk) is 0 or len(term) is 0 or len(subsurf) is 0:
            # print(il, zl)
            # msg = 'dbid: ' + str(atoms.info['id'])
            raise AssertionError()
    except (AssertionError, IndexError):
        layers = int(atoms.info['key_value_pairs']['layers'])
        radii = [get_radius(z)
                 for z in atoms.numbers[atoms.info['slab_atoms']]]
        radius = np.average(radii)
        il, zl = get_layers(atoms, (0, 0, 1), tolerance=radius)
        if len(zl) < layers:
            # msg = 'dbid: ' + str(atoms.info['id'])
            raise AssertionError()
        # bulk atoms includes subsurface atoms.
        bulk = [a.index for a in atoms if il[a.index] <= layers - 2]
        subsurf = [a.index for a in atoms if il[a.index] == layers - 2]
        term = [a.index for a in atoms if il[a.index] >= layers - 1 and
                a.index not in atoms.info['ads_atoms']]
    if len(bulk) is 0 or len(term) is 0 or len(subsurf) is 0:
        print(bulk, term, subsurf)
        msg = 'Detect bulk/term.'
        if 'id' in atoms.info:
            msg += ' id: ' + str(atoms.info['id'])
        print(il, zl)
        raise AssertionError(msg)
    for a_s in atoms.info['site_atoms']:
        if a_s not in term:
            msg = 'site not in term.'
            if 'id' in atoms.info:
                msg += str(atoms.info['id'])
            warnings.warn(msg)
            break
    return bulk, term, subsurf
