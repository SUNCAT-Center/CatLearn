import numpy as np
from sklearn.cluster import KMeans


# Global variables.
transition_metals = ['Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                     'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                     'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg']
metals = ['Li', 'Be', 'Na', 'Mg', 'K', 'Ca', 'Rb', 'Sr', 'Cs', 'Ba', 'Fr',
          'Ra', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm',
          'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Ac', 'Th',
          'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
          'Cf', 'Es', 'Fm', 'Md', 'No', 'Lw', 'B', 'Al', 'Si', 'Ga', 'Ge',
          'In', 'Sn', 'Tl', 'Pb']
metals.extend(transition_metals)


def stoichiometry(atoms):
    """Return a number of layers given a slab.

    Parameters
    ----------
    atoms : object
        ASE atoms object.

    Returns
    -------
    num_dict : dictionary
        First entry is total number of atoms.
        Then key =  element and entry = number
    """
    symbols = atoms.get_chemical_symbols()
    num_dict = {}
    elements = list(set(symbols))
    for element in elements:
        count = symbols.count(element)
        num_dict[element] = count
    return(num_dict)


def is_metal(chemical_symbol):
    """Checks whether string is a metal elementary symbol.

    Parameters
    ----------
    chemical_symbol : string
        The element name.

    Returns
    -------
    metal : Boolean
        Whether it's a metal.
    """
    metal = False
    if chemical_symbol in metals:
        metal = True
    return(metal)


def is_oxide(atoms):
    """Checks whether atms object is an oxide.

    Parameters
    ----------
    atoms : object
        ASE atoms object.

    Returns
    -------
    oxide : Boolean
        Whether it is likely an oxide.
    """
    oxide = False
    num_dict = stoichiometry(atoms)
    elements = list(num_dict.keys())[1:]
    all_count = list(num_dict.values())[0]
    for element in elements:
        if element in transition_metals:
            if num_dict[element]/all_count*1.0 > 0.25:
                try:
                    if num_dict['O']/num_dict[element] >= 1:
                        # print("Materials class likely a metal oxide.")
                        oxide = True
                except KeyError as e:
                    pass
        elif element in metals:
            if num_dict[element]/all_count*1.0 > 0.25:
                try:
                    if num_dict['O']/num_dict[element] >= 1:
                        # print("Materials class likely a metal oxide.")
                        oxide = True
                except KeyError as e:
                    pass
    return(oxide)


def slab_layers(atoms, max_layers=20, tolerance=0.5):
    """Return a number of layers given a slab.

    Parameters
    ----------
    atoms : object
        ASE atoms object.
    max_layers : maximum number of layers expected.
    tolerance : convergence criterion for clustering
    based on the pooled standard deviation of z-coordinates.
    Suggested : 0.5 for oxides, 0.2 for metals.

    Returns
    -------
    layer_avg_z : list
        List of average z-values of all layers.
    layer_atoms : list of list
        Each sublist contains the atom indices of the atoms in that layer.
    """
    oxide = is_oxide(atoms)
    if oxide:
        stoi = stoichiometry(atoms)
        elements = list(stoi.keys())[1:]
        for element in elements:
            if element in metals:
                indices = [atom.index for atom in atoms if
                           atom.symbol == element]
                zpos = [atoms.positions[atom.index][2] for atom in atoms if
                        atom.index in indices]
    else:
        indices = [atom.index for atom in atoms]
        zpos = [atoms.positions[atom.index][2] for atom in atoms]
    X = [[i] for i in zpos]

    for n in range(1, max_layers + 1):
        kmeans = KMeans(n_clusters=n, random_state=0).fit(X)
        clusters = kmeans.labels_
        results = list(zip(indices, zpos, clusters))
        layer_numbers = list(set(clusters))

        # check convergence
        var_i_n_1 = []
        n_i = []
        layer_atoms = []
        layer_avg_z = []
        for i in layer_numbers:
            layer = [entry for entry in results if entry[2] == i]
            layer_atoms.append([entry[0] for entry in layer])
            layer_avg_z.append(np.mean([entry[1] for entry in layer]))
            var_i_n_1.append(np.var([entry[1] for entry in layer]) *
                             (len(layer) - 1))
            n_i.append(len(layer))
        k = len(n_i)
        pooled_standart_deviation = np.sqrt(sum(var_i_n_1) / (sum(n_i) - k))

        if pooled_standart_deviation <= tolerance:
            final_results = list(zip(layer_numbers, layer_avg_z, layer_atoms))
            final_results = sorted(final_results, key=lambda tup: tup[1])
            layer_numbers = list(range(len(final_results)))
            layer_avg_z = [tup[1] for tup in final_results]
            layer_atoms = [tup[2] for tup in final_results]
            # print('Found ' + str(n) + ' layers.')
            return(layer_avg_z, layer_atoms)
