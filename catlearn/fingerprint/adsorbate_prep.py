"""This function constructs a dictionary with abinitio_energies.

Input:
    fname (str) path/filename of ase.db file
    selection (list) ase.db selection
"""
import warnings
import numpy as np
from tqdm import tqdm
from ase.atoms import string2symbols
from ase.geometry import get_layers
from catlearn.api.ase_atoms_api import images_connectivity
from catlearn.fingerprint.periodic_table_data import get_radius


ads_syms = ['H', 'C', 'O', 'N', 'S', 'F', 'Cl', 'P', 'Na', 'K']


def catalysis_hub_to_info(images):
    raise NotImplementedError("Coming soon.")


def autogen_info(images):
    """Return a list of atoms objects with atomic group information
    attached to atoms.subsets.
    This information is  needed by some functions in the
    AdsorbateFingerprintGenerator.

    Parameters
    ----------
    images : list
        list of atoms objects representing adsorbates on slabs.
        No further information is required in atoms.subsets.
    """
    images = images_connectivity(images)
    for i, atoms in tqdm(enumerate(images)):
        # Pre-allocate subsets dictionary.
        if not hasattr(atoms, 'subsets'):
            atoms.subsets = {}
        # Identify adsorbate atoms.
        if 'ads_atoms' not in atoms.subsets:
            ads_atoms = detect_adsorbate(atoms)
            atoms.subsets['ads_atoms'] = ads_atoms
            if len(ads_atoms) == 0:
                continue
        # Identify slab atoms.
        if 'slab_atoms' not in atoms.subsets:
            atoms.subsets['slab_atoms'] = slab_index(atoms)
        # Identify atoms at node distance 0 and 1 from the surface-ads bond.
        if ('chemisorbed_atoms' not in atoms.subsets or
            'site_atoms' not in atoms.subsets or
                'ligand_atoms' not in atoms.subsets):
            chemi, site, ligand = info2primary_index(atoms)
            atoms.subsets['chemisorbed_atoms'] = chemi
            atoms.subsets['site_atoms'] = site
            atoms.subsets['ligand_atoms'] = ligand
        # Identify surface layers.
        if ('bulk_atoms' not in atoms.subsets or
            'termination_atoms' not in atoms.subsets or
           'subsurf_atoms' not in atoms.subsets):
            bulk, term, subsurf = detect_termination(atoms)
            atoms.subsets['bulk_atoms'] = bulk
            atoms.subsets['termination_atoms'] = term
            atoms.subsets['subsurf_atoms'] = subsurf
    return images


def termination_info(images):
    """Return a list of atoms objects with attached information about
    the slab termination, the slab second outermost layer and the bulk slab
    compositions.

    Parameters
    ----------
    images : list
        list of atoms objects representing adsorbates on slabs.
        The atoms objects must have the following keys in atoms.subsets:

            - 'ads_atoms' : list
                indices of atoms belonging to the adsorbate
            - 'slab_atoms' : list
                indices of atoms belonging to the slab
    """
    for atoms in tqdm(images):
        bulk, subsurf, term = detect_termination(atoms)
        atoms.subsets['bulk_atoms'] = bulk
        atoms.subsets['termination_atoms'] = term
        atoms.subsets['subsurf_atoms'] = subsurf
    return images


def detect_adsorbate(atoms):
    """Return a list of indices of atoms belonging to an adsorbate.

    Parameters
    ----------
    atoms : object
        An ase atoms object.
    """
    if 'ads_atoms' in atoms.subsets:
        return atoms.subsets['ads_atoms']
    # Use species chemical formula if known.
    try:
        species = atoms.info['key_value_pairs']['species']
    except KeyError:
        warnings.warn("'species' key missing.")
        return sym2ads_index(atoms)
    if species == '':
        return []
    elif '-' in species or '+' in species or ',' in species:
        raise NotImplementedError('Co-adsorption.')
    try:
        return formula2ads_index(atoms, species)
    except AssertionError:
        return last2ads_index(atoms, species)


def detect_termination(atoms):
    """ Returns three lists, the first containing indices of bulk atoms and
    the second containing indices of atoms in the second outermost layer, and
    the last denotes atoms in the outermost layer or termination or the slab.

    Parameters
    ----------
    atoms : object.
        The atoms object must have the following keys in atoms.subsets:
        'slab_atoms' : list
            indices of atoms belonging to the slab
    """
    if len(np.unique(atoms.get_tags())) >= 4:
        bulk, term, subsurf = tags_termination(atoms)
    elif len(atoms.constraints) == 1:
        bulk, term, subsurf = constraints_termination(atoms)
    elif 'key_value_pairs' not in atoms.info:
        bulk, term, subsurf = connectivity_termination(atoms)
    elif 'layers' in atoms.info['key_value_pairs']:
        bulk, term, subsurf = layers_termination(atoms)
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
    if 'site_atoms' in atoms.subsets:
        for a_s in atoms.subsets['site_atoms']:
            if a_s not in term:
                msg = 'site atom not in term.'
                if 'id' in atoms.info:
                    msg += str(atoms.info['id'])
                warnings.warn(msg)
                break
    return bulk, term, subsurf


def slab_index(atoms):
    """ Returns a list of indices of atoms belonging to the slab.
    These are defined as atoms that are not belonging to the adsorbate.

    Parameters
    ----------
    atoms : ase atoms object
        The atoms object must have the key 'ads_atoms' in atoms.subsets:

            - 'ads_atoms' : list
                indices of atoms belonging to the adsorbate
    """
    chemi = [a.index for a in atoms if a.index not in
             atoms.subsets['ads_atoms']]
    return chemi


def sym2ads_index(atoms):
    """Return the indexes of atoms from the global list of adsorbate symbols.

    Parameters
    ----------
    atoms : object
        An ase atoms object.
    """
    ads_atoms = [a.index for a in atoms if a.symbol in ads_syms]

    return ads_atoms


def last2ads_index(atoms, species):
    """Return the indexes of the last n atoms in the atoms object, where n is
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
    species : str
        chemical formula of the adsorbate.
    """
    n_ads = len(string2symbols(species))
    natoms = len(atoms)
    ads_atoms = list(range(natoms - n_ads, natoms))
    composition = string2symbols(species)
    for a in ads_atoms:
        if atoms[a].symbol not in composition:
            raise AssertionError("last index adsorbate identification failed.")
    return ads_atoms


def formula2ads_index(atoms, species):
    """Return the indexes of atoms, which have symbols matching the chemical
    formula of the adsorbate. This function will not work for adsorbates
    containing the same elements as the slab.

    Parameters
    ----------
    atoms : ase atoms object.
        atoms.info must be a dictionary containing the key 'key_value_pairs',
        which is expected to contain CatMAP standard adsorbate structure
        key value pairs. See the ase db to catmap module in catmap.
        the key value pair 'species' must be the
        chemical formula of the adsorbate.
    species : str
        chemical formula of the adsorbate.
    """
    try:
        composition = string2symbols(species)
    except ValueError:
        print(species)
        raise
    ads_atoms = [a.index for a in atoms if a.symbol in composition]
    if len(ads_atoms) != len(composition):
        raise AssertionError("ads atoms identification by formula failed.")
    return ads_atoms


def layers2ads_index(atoms, species):
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
    species : str
        chemical formula of the adsorbate.
    """
    lz, li = auto_layers(atoms)
    layers = int(atoms.info['key_value_pairs']['layers'])
    ads_atoms = [a.index for a in atoms if li[a.index] > layers - 1]
    natoms = len(atoms)
    composition = string2symbols(species)
    n_ads = len(composition)
    ads_atoms = list(range(natoms - n_ads, natoms))
    if len(ads_atoms) != len(composition):
        raise AssertionError("ads atoms identification by layers failed.")
    return ads_atoms


def z2ads_index(atoms, species):
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
    species : str
        chemical formula of the adsorbate.
    """
    composition = string2symbols(species)
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
        The atoms object must have the following keys in atoms.subsets:

        'ads_atoms' : list
            indices of atoms belonging to the adsorbate
        'slab_atoms' : list
            indices of atoms belonging to the slab
    """
    slab_atoms = atoms.subsets['slab_atoms']
    ads_atoms = atoms.subsets['ads_atoms']
    cm = atoms.connectivity
    chemi = []
    site = []
    ligand = []
    for a_a in ads_atoms:
        for a_s in slab_atoms:
            if cm[a_a, a_s] > 0:
                site.append(a_s)
                chemi.append(a_a)
    chemi = list(np.unique(chemi))
    site = list(np.unique(site))
    for j in site:
        for a_s in slab_atoms:
            if cm[a_s, j] > 0:
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


def tags_termination(atoms):
    """Return lists bulk, term and subsurf containing atom indices belonging to
    those subsets of a surface atoms object. CatKit and ase.build contain
    functions that by default store this information in tags.

    Parameters
    ----------
    atoms : object
        the termination atoms should have tag=1 and subsequent layers should be
        tagged in increasing order.
    """
    term = [a.index for a in atoms if a.tag == 1]
    subsurf = [a.index for a in atoms if a.tag == 2]
    bulk = [a.index for a in atoms if a.tag > 1]
    return bulk, term, subsurf


def layers_termination(atoms):
    """Return lists bulk, term and subsurf containing atom indices belonging to
    those subsets of a surface atoms object.
    This function relies on ase.atoms.get_layers, default atomic radii, and a
    slab oriented in the xy plane, where the termination in the z+ direction
    is the surface.

    Parameters
    ----------
    atoms : object
        The atoms object must have the following keys in atoms.subsets:

        'slab_atoms' : list
            indices of atoms belonging to the slab.
    """
    il, zl = auto_layers(atoms)
    il_slab = list(il)
    for index in sorted(atoms.subsets['ads_atoms'], reverse=True):
        del il_slab[index]
    nlayers = len(np.unique(il_slab))
    try:
        db_layers = atoms.info['key_value_pairs']['layers']
        if db_layers != nlayers:
            msg = str(db_layers) + ' != ' + str(nlayers)
            if 'id' in atoms.info:
                msg += ' ' + str(atoms.info['id'])
            warnings.warn(msg)
    except(KeyError):
        if nlayers < 3:
            msg = 'nlayers = ' + str(nlayers)
            if 'id' in atoms.info:
                msg += ' ' + str(atoms.info['id'])
            warnings.warn(msg)
    # bulk atoms can include subsurface atoms.
    term = [a.index for a in atoms if il[a.index] >= nlayers - 1 and
            a.index not in atoms.subsets['ads_atoms']]
    bulk = [a.index for a in atoms if il[a.index] <= nlayers - 2]
    subsurf = [a.index for a in atoms if il[a.index] == nlayers - 2]
    return bulk, term, subsurf


def constraints_termination(atoms):
    """Return lists bulk, term and subsurf containing atom indices belonging to
    those subsets of a surface atoms object.
    This function relies on the connectivity of the atoms and it assumes that
    bulk atoms are those that have are constrained in the first constraint.

    Parameters
    ----------
    atoms : object
        atoms.connectivity should be a connectivity matrix.
        The atoms object must have the following keys in atoms.subsets:

        'slab_atoms' : list
            indices of atoms belonging to the slab.
    """
    cm = atoms.connectivity.copy()
    np.fill_diagonal(cm, 0)
    # List coordination numbers of slab atoms.
    coord = np.count_nonzero(cm[atoms.subsets['slab_atoms'], :], axis=1)
    coord -= 1
    bulk = []
    term = []
    max_coord = np.max(coord)
    for j, c in enumerate(coord):
        a_s = atoms.subsets['slab_atoms'][j]
        if c == max_coord:
            bulk.append(a_s)
        elif a_s in atoms.constraints[0].get_indices():
            bulk.append(a_s)
        else:
            term.append(a_s)
    subsurf = []
    for a_b in bulk:
        for a_t in term:
            if cm[a_t, a_b] > 0:
                subsurf.append(a_b)
    subsurf = list(np.unique(subsurf))
    return bulk, term, subsurf


def connectivity_termination(atoms):
    """Return lists bulk, term and subsurf containing atom indices belonging to
    those subsets of a surface atoms object.
    This function relies on the connectivity of the atoms.

    Parameters
    ----------
    atoms : object
        atoms.connectivity should be a connectivity matrix.
        The atoms object must have the following keys in atoms.subsets:

        'slab_atoms' : list
            indices of atoms belonging to the slab
    """
    raise NotImplementedError("Todo: identify and remove backside termination")
    cm = atoms.connectivity.copy()
    np.fill_diagonal(cm, 0)
    # List coordination numbers of slab atoms.
    coord = np.count_nonzero(cm[atoms.subsets['slab_atoms'], :], axis=1)
    coord -= 1
    bulk = []
    term = []
    max_coord = np.max(coord)
    bulk = atoms.subsets['slab_atoms']
    for j, c in enumerate(coord):
        a_s = atoms.subsets['slab_atoms'][j]
        if c == max_coord:
            bulk.append(a_s)
        else:
            term.append(a_s)
    subsurf = []
    for a_b in bulk:
        for a_t in term:
            if cm[a_t, a_b] > 0:
                subsurf.append(a_b)
    subsurf = list(np.unique(subsurf))
    return bulk, term, subsurf


def auto_layers(atoms):
    """Returns two arrays describing which layer each atom belongs
    to and the distance between the layers and origo.
    Assumes the tolerance corresponds to the average atomic radii of the slab.

    Parameters
    ----------
    atoms : object
        The atoms object must have the following keys in atoms.subsets:

        'slab_atoms' : list
            indices of atoms belonging to the slab
    """
    radii = [get_radius(z) for z in atoms.numbers[atoms.subsets['slab_atoms']]]
    radius = np.average(radii)
    lz, li = get_layers(atoms, (0, 0, 1), tolerance=radius)
    return lz, li
