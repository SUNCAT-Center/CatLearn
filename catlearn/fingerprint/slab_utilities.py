import ase
import catkit
import json
import numpy as np
import os
import sklearn
import sys

from ase.build import bulk
from ase.constraints import *
from ase.visualize import view
from catkit.build import molecule
from catkit.gen.adsorption import Builder
from catkit.gen.surface import SlabGenerator
from sklearn.cluster import KMeans

# global variables
transition_metals = ['Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                     'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                     'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg']
metals = ['Li', 'Be', 'Na', 'Mg', 'K', 'Ca', 'Rb', 'Sr', 'Cs', 'Ba', 'Fr', 'Ra', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm',
          'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
          'Cf', 'Es', 'Fm', 'Md', 'No', 'Lw', 'B', 'Al', 'Si', 'Ga', 'Ge', 'In', 'Sn', 'Tl', 'Pb']
metals.extend(transition_metals)

def auto_kpoints(atoms, effective_length=25, bulk=False):
    """Return a tuple of k-points derived from the unit cell.

    Parameters
    ----------
        atoms : object
        effective_length : k-point*unit-cell-parameter
        bulk : Whether it is a bulk system.
    """
    cell = atoms.get_cell()
    nkx = int(round(effective_length/np.linalg.norm(cell[0]),0))
    nky = int(round(effective_length/np.linalg.norm(cell[1]),0))
    if bulk == True:
        nkz = int(round(effective_length/np.linalg.norm(cell[2]),0))
    else:
        nkz = 1
    return(nkx, nky, nkz)

def fix_bottom_layers(atoms, top_layers=1):
    """Return a atoms object with bottom layers fixed
    Parameters
    ----------
        atoms : object
        top_layers : number of layers to keep free
    """
    # atom_idx is a list of lists containing the atoms in the layer
    layer_z, atom_idx = slab_layers(atoms)
    fixed_atoms = atom_idx[:-1]
    fixed_atoms = [val for sublist in fixed_atoms for val in sublist]
    c = FixAtoms(indices = fixed_atoms)
    atoms.set_constraint([c])
    return(atoms)

def is_metal(chemical_symbol):
    """Checks whether string is a metal elementary symbol.
    Parameters
    ----------
        chemical_symbol : string
    """
    metal = False
    if chemical_symbol in metals:
        metal = True
    return(metal)

def is_oxide(atoms):
    """Checks whether a system might be an oxide.
    Parameters
    ----------
        atoms : object.
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
                        # print("Materials class likely a transition metal oxide.")
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

def lower_slab(atoms):
    """Lowers the slab in the unit cell,
    so that the bottom layer is at zero.
    Parameters
    ----------
        atoms : object.
    """
    z = []
    for atom in atoms:
        z.append(atom.position[2])
    z = sorted(z)
    for atom in atoms:
        atoms.positions[atom.index][2] = atoms.positions[atom.index][2] - z[0]
    return(atoms)

def save_structure(atoms, slab_element=None, facet=None, adsorbate=None, local_path=None):
    """ Save the structure and automatically find out where.

    Parameters
    ----------
        atoms : object
        slab_element : string of slab name or element
        facet : Miller indices of surface
        adsorbate : strin of adsorbate name
        local path : costumize your folder name
    """
    if slab_element and facet and adsorbate:
        traj = adsorbate+'_'+slab_element+str(facet)+'.json'
        folder = os.getcwd()+'/'+slab_element+'/'+str(facet)+'/'+adsorbate+'/'
    elif slab_element and facet and not adsorbate:
        traj = slab_element+str(facet)+'.json'
        folder = os.getcwd()+'/'+slab_element+'/'+str(facet)+'/clean/'
    else:
       traj = "".join(list(set(atoms.get_chemical_symbols())))+'.json'
       folder = os.getcwd()+'/'+local_path
    print(folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
    ase.io.write(folder+traj,atoms)
    return(atoms)

def setbox(atoms, x=None, y=None, z=21):
    """Sets the unit cell of the system.
    Parameters
    ----------
        atoms : object.
    """
    cell = atoms.get_cell()
    if not x and not y:
        atoms.set_cell([cell[0],cell[1],[0,0,z]], scale_atoms=False)
    else:
        atoms.set_cell([[x,0,0], [0,y,0],[0,0,z]], scale_atoms=False)
    return(atoms)

def slab_layers(atoms, max_layers=20, tolerance=0.5):
    """Return a number of layers given a slab.
    Parameters
    ----------
        atoms : object.
        max_layers : maximum number of layers expected.
        tolerance : convergence criterion for clustering
        based on the pooled standard deviation of z-coordinates.
    """
    oxide = is_oxide(atoms)
    if oxide:
        stoichiometry = stoichiometry(atoms)
        elements = list(stoichiometry.keys())[1:]
        for element in elements:
            if element in metals:
                indices = [atom.index for atom in atoms if atom.symbol == element]
                zpos = [atoms.positions[atom.index][2] for atom in atoms if atom.index in indices]
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
            var_i_n_1.append(np.var([entry[1] for entry in layer]) * (len(layer) - 1))
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

def stoichiometry(atoms):
    """Gives the stoichiometry of the system in form of a dictionary.
    Parameters
    ----------
        atoms : object.
    """
    symbols = atoms.get_chemical_symbols()
    num_dict = {}
    num_dict['total'] = len(symbols)
    elements = list(set(symbols))
    for element in elements:
        count = symbols.count(element)
        num_dict[element] = count
    return(num_dict)
