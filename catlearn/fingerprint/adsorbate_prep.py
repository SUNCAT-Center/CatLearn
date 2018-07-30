"""This function constructs a dictionary with abinitio_energies.

Input:
    fname (str) path/filename of ase.db file
    selection (list) ase.db selection
"""
import warnings
import numpy as np
from tqdm import tqdm
from ase.data import chemical_symbols
from ase.atoms import string2symbols
# get_distances requires ASE 3.16 or above.
from ase.geometry import get_layers, get_distances
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
    if (len(np.unique(atoms.get_tags())) >= 4 and
        min(atoms.get_tags()) <= -1 and
       max(atoms.get_tags()) > 2):
        return tags2ads_index(atoms)
    # Use species chemical formula if known.
    try:
        species = atoms.info['key_value_pairs']['species']
    except KeyError:
        warnings.warn("'species' key or .info['key_value_pairs'] missing.")
        return sym2ads_index(atoms, ads_syms=ads_syms)
    if species == '':
        return []
    elif '-' in species or '+' in species or ',' in species:
        raise NotImplementedError('Co-adsorption.')
    try:
        return formula2ads_index(atoms, species)
    except AssertionError:
        try:
            return connectivity2ads_index(atoms, species)
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
    if (len(np.unique(atoms.get_tags())) >= 4 and
        min(atoms.get_tags()) <= 0 and
       max(atoms.get_tags()) > 2):
        bulk, term, subsurf = tags_termination(atoms)
    elif len(atoms.constraints) == 1:
        bulk, term, subsurf = constraints_termination(atoms)
    elif 'key_value_pairs' not in atoms.info:
        bulk, term, subsurf = connectivity_termination(atoms)
    elif 'layers' in atoms.info['key_value_pairs']:
        bulk, term, subsurf = layers_termination(atoms)
    else:
        bulk, term, subsurf = connectivity_termination(atoms)
    try:
        sx, sy = atoms.info['key_value_pairs']['supercell'].split('x')
        sx = int(''.join(i for i in sx if i.isdigit()))
        sy = int(''.join(i for i in sy if i.isdigit()))
        if int(sx) * int(sy) != len(term):
            msg = str(len(term)) + ' termination atoms identified.' + \
                ' ' + atoms.info['key_value_pairs']['facet'] + \
                ' ' + atoms.info['key_value_pairs']['supercell'] + \
                '. id=' + str(atoms.info['id'])
            warnings.warn(msg)
    except KeyError:
        pass
    if 'site_atoms' in atoms.subsets:
        for a_s in atoms.subsets['site_atoms']:
            if a_s not in term:
                msg = 'site atom not in term.'
                try:
                    msg += ' ' + str(atoms.info['key_value_pairs']['phase'])
                except KeyError:
                    pass
                try:
                    msg += ' ' + str(atoms.info['key_value_pairs']['facet'])
                    msg += ' ' + str(
                            atoms.info['key_value_pairs']['supercell'])
                except KeyError:
                    pass
                if 'id' in atoms.info:
                    msg += ' ' + str(atoms.info['id'])
                warnings.warn(msg)
                break
    return bulk, term, subsurf


def check_reconstructions(image_pairs):
    """Return a list of database ids, for adsorbate/slab structures, which
    has a reconstructed slab with respect to the reference slab.

    Parameters
    ----------
    image_pairs : list
        List of tuples containing pairs of ASE atoms objects.
        The first element in each tuple must represent an adsorbate*slab
        structure and the second element must represent a slab.
    """
    reconstructed = []
    for row in image_pairs:
        identical = compare_slab_connectivity(row[0], row[1])
        if not identical:
            reconstructed.append(row[0].info['id'])
    return reconstructed


def compare_slab_connectivity(atoms, reference_atoms):
    """Return a boolean for whether an adsorbate has caused a slab to
    reconstruct and change it's connectivity.

    Parameters
    ----------
    atoms : object
        ASE atoms object with connectivity and 'slab_atoms' subsets attached.
        This represents an adsorbate*slab structure.
    reference_atoms : object
        ASE atoms object with connectivity and 'slab_atoms' subsets attached.
        This represents a slab structure.

    Returns
    ----------
    identical : boolean
        Are the connectivities within the slabs identical or not.
    """
    slab_atoms = atoms.subsets['slab_atoms']
    slab_cm = atoms.connectivity[:, slab_atoms][slab_atoms, :]
    reference_cm = reference_atoms.connectivity
    return np.allclose(slab_cm, reference_cm)


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


def sym2ads_index(atoms, ads_syms):
    """Return the indexes of atoms from the global list of adsorbate symbols.

    Parameters
    ----------
    atoms : object
        An ase atoms object.
    """
    ads_atoms = [a.index for a in atoms if a.symbol in ads_syms]

    return ads_atoms


def connectivity2ads_index(atoms, species):
    """Return the indexes of atoms from the global list of adsorbate symbols.

    Parameters
    ----------
    atoms : object
        ASE atoms object with connectivity attached.
        This represents an adsorbate*slab structure.
    species : str
        chemical formula of the adsorbate.
    """
    composition = string2symbols(species)
    ads_atoms = []
    for symbol in composition:
        if (composition.count(symbol) ==
           atoms.get_chemical_symbols().count(symbol)):
            ads_atoms += [atom.index for atom in atoms if
                          atom.symbol == symbol]

    if len(ads_atoms) == len(composition):
        return ads_atoms
    elif len(ads_atoms) == 0:
        raise AssertionError("Formula adsorbate identification failed.")

    # If an atom in species also occurs in the slab, infer by connectivity.
    connected_atoms = []
    for atom in ads_atoms:
        edges = atoms.connectivity[atom, :]
        connected_atoms += [i for i, bonds in enumerate(edges) if
                            bonds > 0 and
                            atoms[i].symbol in composition and
                            atoms[i].symbol != atoms[atom].symbol]
    ads_atoms += connected_atoms

    # Final check.
    ua_ads, uc_ads = np.unique(
        np.array(atoms.get_chemical_symbols())[ads_atoms].sort(),
        return_counts=True)
    ua_comp, uc_comp = np.unique(composition.sort(), return_counts=True)
    if ua_ads != ua_comp:
        msg = str(ua_ads) + " != " + str(ua_comp)
        raise AssertionError(msg)
    elif uc_ads != uc_comp:
        msg = str(uc_ads) + " != " + str(uc_comp)
        raise AssertionError(msg)

    return list(np.unique(ads_atoms))


def slab_positions2ads_index(atoms, slab, species):
    """Return the indexes of adsorbate atoms identified by comparing positions
    to a reference slab structure.

    Parameters
    ----------
    atoms : object
    """
    composition = string2symbols(species)
    ads_atoms = []
    for symbol in composition:
        if (composition.count(symbol) ==
           atoms.get_chemical_symbols().count(symbol)):
            ads_atoms += [atom.index for atom in atoms if
                          atom.symbol == symbol]

    ua_ads, uc_ads = np.unique(ads_atoms, return_counts=True)
    ua_comp, uc_comp = np.unique(composition, return_counts=True)
    if ua_ads == ua_comp and uc_ads == uc_comp:
        return ads_atoms

    p_a = atoms.get_positions()
    p_r = slab.get_positions()

    for s in composition:
        if s in np.array(atoms.get_chemical_symbols())[ads_atoms]:
            continue
        symbol_count = composition.count(s)
        index_a = np.where(np.array(atoms.get_chemical_symbols()) == s)[0]
        index_r = np.where(np.array(slab.get_chemical_symbols()) == s)[0]
        _, dist = get_distances(p_a[index_a, :], p2=p_r[index_r, :],
                                cell=atoms.cell, pbc=True)
        # Assume all slab atoms are closest to their reference counterpart.
        deviations = np.min(dist, axis=1)
        # Sort deviations.
        ascending = np.argsort(deviations)
        # The highest deviations are assumed to be new atoms.
        ads_atoms += list(index_a[ascending[-symbol_count:]])

    # Final check.
    ua_ads, uc_ads = np.unique(
        np.array(atoms.get_chemical_symbols())[ads_atoms].sort(),
        return_counts=True)
    ua_comp, uc_comp = np.unique(composition.sort(), return_counts=True)
    if ua_ads != ua_comp:
        msg = str(ua_ads) + " != " + str(ua_comp)
        raise AssertionError(msg)
    elif uc_ads != uc_comp:
        msg = str(uc_ads) + " != " + str(uc_comp)
        raise AssertionError(msg)

    return ads_atoms


def tags2ads_index(atoms):
    """Return the indexes of atoms from the global list of adsorbate symbols.

    Parameters
    ----------
    atoms : object
        An ase atoms object. `atoms.tags` must label adsorbate atoms with 0 or
        negative numbers.
    """
    ads_atoms = [a.index for a in atoms if a.tag < 0]

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
    warnings.warn("Adsorbate identified by last index.")
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
            msg += ' id=' + str(atoms.info['id'])
        warnings.warn(msg)
    elif len(ligand) < 2:
        msg = 'Site has less than 2 nearest neighbors.'
        if 'id' in atoms.info:
            msg += ' id=' + str(atoms.info['id'])
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


def layers_termination(atoms, miller=(0, 0, 1)):
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
    il, zl = auto_layers(atoms, miller=miller)
    il_slab = list(il)
    for index in sorted(atoms.subsets['ads_atoms'], reverse=True):
        del il_slab[index]
    nlayers = len(np.unique(il_slab))
    try:
        db_layers = atoms.info['key_value_pairs']['layers']
        if db_layers != nlayers:
            msg = str(db_layers) + ' != ' + str(nlayers)
            if 'id' in atoms.info:
                msg += ' id=' + str(atoms.info['id'])
            warnings.warn(msg)
    except(KeyError):
        if nlayers < 3:
            msg = 'nlayers = ' + str(nlayers)
            if 'id' in atoms.info:
                msg += ' id=' + str(atoms.info['id'])
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
    slab_atoms = atoms.subsets['slab_atoms']
    coord = np.count_nonzero(cm[slab_atoms, :][:, slab_atoms], axis=1)
    bulk = []
    term = []
    max_coord = np.max(coord)
    for j, c in enumerate(coord):
        a_s = atoms.subsets['slab_atoms'][j]
        if a_s in atoms.constraints[0].get_indices():
            bulk.append(a_s)
        elif c >= max_coord - 2:
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
    slab_atoms = atoms.subsets['slab_atoms']
    coord = np.count_nonzero(cm[slab_atoms, :][:, slab_atoms], axis=1)
    bulk = []
    term = []
    max_coord = np.max(coord)
    bulk = atoms.subsets['slab_atoms']
    for j, c in enumerate(coord):
        a_s = atoms.subsets['slab_atoms'][j]
        if c >= max_coord - 2:
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


def auto_layers(atoms, miller=(0, 0, 1)):
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
    radius = np.average(radii) / 2.
    lz, li = get_layers(atoms, miller=miller, tolerance=radius)
    return lz, li


def attach_cations(atoms, anion_number=8):
    if anion_number not in atoms.numbers:
        raise ValueError('Anion ' + chemical_symbols[anion_number] +
                         'not in atoms')
    atoms.subsets['cation_atoms'] = [a.index for a in atoms if
                                     a.number != anion_number]
    atoms.subsets['anion_atoms'] = [a.index for a in atoms if
                                    a.number == anion_number]
