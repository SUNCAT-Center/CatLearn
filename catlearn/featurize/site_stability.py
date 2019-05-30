import os
import sys
cwd = os.getcwd()+'/'
sys.path.insert(0, cwd+'CatLearn/')

import ase.io
import CatLearn.catlearn as catlearn
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
import re
import seaborn as sns

from ase.calculators.emt import EMT

from catlearn.featurize import *
from catlearn.featurize.slab_utilities import *

from itertools import chain, combinations
from mendeleev import element

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from tqdm import tqdm

# pd.set_option('display.max_columns', 10)

# GLOBBALS AND HELPER FUNCTIONS
font = {'size': 16}
matplotlib.rc('font', **font)
sns.set_style('white')
sns.set_palette(sns.hls_palette(8, h=0.6, l=0.6, s=0.6))

infile = ''

atom_dict ={'Ni': -1380.8341027932559,
 'Au': -1174.461466369728,
 'Pd': -1083.667223896331,
 'Pt': -998.5654412676529,
 'Ag': -1306.3829300840296,
 'Cu': -1611.535995452583}

def composition_name(atoms):
    """
    Takes atoms and returns stoichiometric composition names without duplicate elements.
    :param atoms: Atoms object.
    :return: str composition name.
    """
    itm = list(stoichiometry(atoms).items())
    itm = [str(e) for l in itm for e in l]
    return ''.join(itm)

def traj_to_reference_dict(folderpath):
    """
    Takes absolute folder path with trajectories of reference atoms
    and molecules and transforms it to an energy dictionary.
    :param folderpath: Path to trajectories in .traj or .json.
    :return: dict: Reference dictionary.
    """
    refDict = {}
    for f in os.listdir(folderpath):
        fp = folderpath
        if not fp.endswith('/'):
            fp = fp + '/'
        if f.endswith('traj') or f.endswith('json'):
            atoms = ase.io.read(fp+f)
            refDict[str(atoms.symbols)] =atoms.get_potential_energy()
    return refDict


def get_df(filename):
    """
    Read in absolute file path to tsv, returns dataframe wihtout index column.
    :param filename: Full path to file.
    :return: pd.DataFrame.
    """
    df = pd.read_csv(filename, sep='\t', index_col=False)
    df = df.rename(columns={df.columns[0]: 'INDEX'})
    df = df.drop(['INDEX'], axis=1)
    return df


def update_site_file(infile_pickle, infile_energy):
    """
    Takes pickle file of SiteFeaturizer object pickle and a simple energy.tsv
    file containing site description like 'material4_site6' and energy.
    :param infile_pickle: Full path to SiteFeaturizer sites pickle file.
    :param infile_energy: Full path energy file.
    :return: List of dictionaries (SiteFeaturizer sites).
    """
    pp = infile_pickle
    outpath = '/'.join(pp.split('/')[:-1]) + '/'
    ep = infile_energy

    # Get site identity and energy
    df = get_df(ep)
    df['sys'] = df.cluster.str.split('_')
    df['material_image_index'] = df.sys.apply(lambda row: int(re.findall('\d+', row[0])[0]))
    df['site_index'] = df.sys.apply(lambda row: int(re.findall('\d+', row[1])[0]))

    # Get the sites dictionaries
    rs = pickle.load(open(pp, "rb"))

    # Update defect total energies
    for idx, row in enumerate(df.material_image_index):
        for i, rowdict in enumerate(rs):
            if rowdict['material_image_index'] == row and rowdict['site_index'] == df.site_index[idx]:
                material = Material(rowdict['material'].atoms)
                defect = Material(rowdict['defect'].atoms)
                defect._total_energy = float(df.energy[idx])
                _ = defect.cohesive_energy
                rs[i]['defect'] = defect
                rs[i]['material'] = material

                # Save results
    f = 'site_selection_' + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '.p'
    pickle.dump(rs, open(outpath + f, "wb"))
    return rs


def get_site_index(material, defect):
    """
    Given two trajectories with equal atom positions and one atom difference,
    determines the site index of the defect site.
    :param material:
    :param defect:
    :return:
    """
    matlist = material.get_positions()
    deflist = defect.get_positions()
    site_detected = []
    for pos in matlist:
        boollist = [np.allclose(pos, defpos, rtol=1e-03) for defpos in deflist]
        site_detected.append(any(boollist))
    site_idx = [idx for idx, _ in enumerate(site_detected) if not _]
    if len(site_idx) == 0:
        site_idx = [np.NAN]
    return site_idx[0]


def unique_set(iterable, feature_dim=2):
    ''' Find unique sets of n descriptors '''
    s = list(iterable)  # allows duplicate elements
    return chain.from_iterable(combinations(s, r) for r in range(feature_dim, feature_dim + 1))


class Material:
    """Material object. Stores and computes atomic system properties, site properties."""
    def __init__(self, atoms, reference_dict=atom_dict):
        """The Site class defines an atomic site in an Atoms object.
        Args:
            atoms (obj) : Atoms object

        Attributes:
            site_index : Atom index of the site to be featurized.
            atoms : The atoms object that the site belongs to.
            cm : Coordination matrix.
            cn_list = list of coordination numbers sorted by atomic index.
            cn : Coordination number of the site.
            avg_neighbor_cn : average coordination number of neighbors.
        """
        self.atoms = atoms
        self.reference_dict = reference_dict
        self.atomic_number_list = self.atoms.numbers
        self.atomic_numbers = list(set(self.atomic_number_list))
        self.cm = neighbor_matrix.connection_matrix(atoms, dx=2.0)
        self.cn_list = np.sum(self.cm, axis=1)
        self.composition = stoichiometry(self.atoms)
        self.composition_coefficients = list(self.composition.values())
        self.natoms = float(len(self.atomic_number_list))
        self.symbols = list(self.composition.keys())
        self.selected_sites = None
        self._total_energy = None
        self._cohesive_energy = None

        self.site_index = None
        self.site_symbol = None
        self.site_atomic_number = None
        self.site_e_val = None
        self.site_cn = None

        self.max_neighbor_cn = None
        self.min_neighbor_cn = None
        self.avg_neighbor_cn = None

        self.site_cm = None
        self.max_neighbor_atomic_number = None
        self.min_neighbor_atomic_number = None
        self.avg_neighbor_atomic_number = None

        self.site_features = None

        self._site_angles_min = None
        self._site_angles_max = None
        self._site_angles_mean = None
        self._site_angles_std = None
        self._site_angles = None

        self._site_distances = None
        self._site_distance_mean = None
        self._site_distance_std = None

        self._system_features = None

    @property
    def total_energy(self):
        """Gets the total energy from atoms object.
        Make sure the trajectory or atoms object has energy."""
        if self._total_energy is None:
            try:
                self._total_energy = float(self.atoms.get_potential_energy())  # eV
            except:
                self._total_energy = None
        return self._total_energy

    @property
    def cohesive_energy(self):
        """Computes the cohesive energy of the whole atoms system."""
        if self._cohesive_energy is None:
            c = self.total_energy
            for key, value in self.composition.items():
                c -= (self.reference_dict[key] * value)
            c = c / self.natoms
            self._cohesive_energy = c
        return self._cohesive_energy

    def get_features(self, site_index, atomic_symbol=False, use_EMT=False):
        """
        Computes system-specific and site-specific features.
        :param site_index: Atomic index of site to be featurized.
        :param atomic_symbol: Whether to use atomic symbols as features.
        :param use_EMT: Whether to use EMT-derived site stabilities as features.
        :return: pd.DataFrame with features.
        """
        site_features = self.get_site_features(site_index=site_index,
                                               atomic_symbol=atomic_symbol,
                                               use_EMT=use_EMT)
        system_features = self.system_features
        df = pd.concat([site_features, system_features], axis=1)
        return df

    def get_site_features(self, site_index, atomic_symbol=False, use_EMT=False):
        """
        Computes site-specific features.
        :param site_index: Atomic index of site to be featurized.
        :param atomic_symbol: Whether to use atomic symbols as features.
        :param use_EMT: Whether to use EMT-derived site stabilities as features.
        :return: pd.DataFrame with features.
        """
        self.site_index = site_index
        self.site_symbol = self.atoms[self.site_index].symbol
        self.site_atomic_number = element(self.site_symbol).atomic_number
        self.site_e_val = element(self.site_symbol).nvalence()
        self.site_cn = self.cn_list[site_index]
        all_atoms_cn = self.cn_list.copy()

        # Neighbor coordination
        neighbor_cn = all_atoms_cn * self.cm[site_index]
        neighbor_cn = np.ma.masked_equal(neighbor_cn, 0)
        self.max_neighbor_cn = neighbor_cn.max()
        self.min_neighbor_cn = neighbor_cn.min()
        self.avg_neighbor_cn = neighbor_cn.mean()

        # Neighbor atomic number
        self.site_cm = self.cm[site_index]
        neighbor_atomic_numbers = self.atomic_number_list * self.cm[site_index]
        neighbor_atomic_numbers = np.ma.masked_equal(neighbor_atomic_numbers, 0)
        self.max_neighbor_atomic_number = neighbor_atomic_numbers.max()
        self.min_neighbor_atomic_number = neighbor_atomic_numbers.min()
        self.avg_neighbor_atomic_number = neighbor_atomic_numbers.mean()

        # EMT site stability
        if use_EMT:
            _ = self.get_EMT_site_stability(site_index=self.site_index)

        # Site atoms triad angles and pair distances
        _ = self.site_angles
        _ = self.site_distances

        # Gather all into feature list
        column_names = ['site_index', 'site_atomic_number', 'site_e_val', 'site_cn',
                        'max_neighbor_cn', 'min_neighbor_cn',
                        'avg_neighbor_cn', 'max_neighbor_atomic_number',
                        'min_neighbor_atomic_number', 'avg_neighbor_atomic_number',
                        'site_distance_mean', 'site_distance_std',
                        'site_angles_min', 'site_angles_max', 'site_angles_mean', 'site_angles_std']
        if atomic_symbol:
            column_names = ['symbol'] + column_names
        if use_EMT:
            column_names = ['EMT_site_stability'] + column_names

        site_features = [self.site_index, self.site_atomic_number, self.site_e_val, self.site_cn,
                         self.max_neighbor_cn, self.min_neighbor_cn,
                         self.avg_neighbor_cn, self.max_neighbor_atomic_number,
                         self.min_neighbor_atomic_number, self.avg_neighbor_atomic_number,
                         self._site_distance_mean, self._site_distance_std,
                         self._site_angles_min, self._site_angles_max,
                         self._site_angles_mean, self._site_angles_std]
        if atomic_symbol:
            site_features.insert(0, self.site_symbol)
        if use_EMT:
            site_features.insert(0, self.EMT_site_stability)

        df = pd.DataFrame([site_features], columns=column_names)
        self.site_features = df
        return self.site_features

    def _unique_set(self, iterable, feature_dim=2):
        """Find unique sets of n features.
        :param iterable: List or other iterable.
        :param feature_dim: dimension of final combinations.
        :return: List of lists of combinations.
        """
        s = list(iterable)  # allows duplicate elements
        c = chain.from_iterable(combinations(s, r) for r in range(feature_dim, feature_dim + 1))
        sn = [x for x in c]
        return sn

    @property
    def site_angles(self):
        """Computes angles of unique triples of atoms with the site being the middle atom.
        Computes statistics from angles.
        :return: List of features."""
        cm_site = self.cm[self.site_index]
        cm_site[self.site_index] = 0
        neighbors = [i for i, x in enumerate(cm_site) if x == 1]
        pairs = self._unique_set(iterable=neighbors)

        pos = self.atoms.get_positions()
        b = pos[self.site_index]

        angles = []
        for pair in pairs:
            a = pos[pair[0]]
            c = pos[pair[1]]
            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)
            angles.append(np.degrees(angle))

        self._site_angles = np.array(angles)
        try:
            self._site_angles_min = np.min(self._site_angles)
            self._site_angles_max = np.max(self._site_angles)
            self._site_angles_mean = np.mean(self._site_angles)
            self._site_angles_std = np.std(self._site_angles)
        except:
            print('This system has angle problems: %s' % str(self.composition))
            self._site_angles_min = np.NAN
            self._site_angles_max = np.NAN
            self._site_angles_mean = np.NAN
            self._site_angles_std = np.NAN
        if len(self._site_angles) == 0:
            self._site_angles = np.NAN

        return self._site_angles

    @property
    def site_distances(self):
        """
        Computes site distances to neighboring atoms.
        :return: List of features.
        """
        cm_site = self.cm[self.site_index]
        cm_site[self.site_index] = 0
        neighbors = [i for i, x in enumerate(cm_site) if x == 1]

        pos = self.atoms.get_positions()
        b = pos[self.site_index]

        distances = []
        for atom_idx in neighbors:
            a = pos[atom_idx]
            ba = np.linalg.norm(a - b)
            distances.append(ba)

        self._site_distances = np.array(distances)
        self._site_distance_mean = np.mean(self._site_distances)
        self._site_distance_std = np.std(self._site_distances)
        return self._site_distances

    @property
    def system_features(self):
        """Computes system-specific features.
        :return: pd.DataFrame with features."""
        df = pd.DataFrame()

        # general
        df['natoms'] = self.natoms
        df['DFT_cohesive_energy'] = self.cohesive_energy

        # atom-wise
        ans = []
        evals = []
        for i, el in enumerate(self.composition.keys()):
            elid = element(el).atomic_number
            ans.append(elid)
            e_val = element(el).nvalence()
            evals.append(e_val)
            df['Z' + str(i)] = [elid]
            df['n' + str(i)] = self.composition[el]
            df['e_val' + str(i)] = [e_val]

        # mean
        df['Z_mean'] = np.mean(ans)
        df['e_val_mean'] = np.mean(evals)

        self._system_features = df
        return self._system_features

    def create_defect(self, site_index):
        """
        Creates an atom object from initial material with defect at site.
        :param site_index: Index of specified site.
        :return: Atoms object.
        """
        self.site_index = site_index
        defect = self.atoms.copy()
        del defect[self.site_index]
        return defect

    def get_EMT_site_stability(self, site_index):
        """
        Returns EMT site stability for selected metals.
        :param int site_index: Atomic number index of the site.
        :return: EMT site stability
        """
        self.site_index = site_index
        self.site_symbol = self.atoms[self.site_index].symbol

        # Atoms energy
        atoms = self.atoms.copy()
        atoms.set_calculator(EMT())
        e_atoms = atoms.get_potential_energy()

        # Site atom vacuum energy
        site = ase.Atoms(self.site_symbol)
        site.set_calculator(EMT())
        e_site_atom = site.get_potential_energy()

        # Defect atoms
        defect = self.create_defect(site_index=self.site_index)
        defect.set_calculator(EMT())
        e_atoms_defect = defect.get_potential_energy()

        self.EMT_site_stability = e_atoms - e_site_atom - e_atoms_defect
        return self.EMT_site_stability


class SiteFeaturizer():
    """Class to handle data set of sites, including atoms,
    reference energies, features and site generation, outlier removal."""
    def __init__(self, images=None,
                 sites=None,
                 use_atomic_symbols=False,
                 use_EMT=False,
                 reference_dict=atom_dict):
        """
        Instantiates the SiteFeaturizer class with:
        :param images: List of Atom objects, optional.
        :param sites: List of sites, such as those created by the class.
        :param use_atomic_symbols: Whether to use atomic symbols as features.
        :param use_EMT: Whether to use EMT site stabilities as features (metals only).
        :param reference_dict: Dictionary of reference energies (atoms or molecule energies).
        """
        self.images = images
        self.sites = sites
        self.use_atomic_symbols = use_atomic_symbols
        self.use_EMT = use_EMT
        self.reference_dict = reference_dict

        self._features = None
        self._site_features = None
        self._normalized_site_features = None

    def update_materials(self):
        """Re-instantiates sites from material and defect Atom objects
        and their total energies."""
        for i, site in enumerate(self.sites):
            # update material
            newmat = Material(atoms=site['material'].atoms,
                              reference_dict=self.reference_dict)


            # update defect
            newdef = Material(atoms=site['defect'].atoms,
                              reference_dict=self.reference_dict)

            # check if all energies are available
            if site['material'].total_energy is None or site['defect'].total_energy is None:
                print('Skipping site: '+str(i))
            else:
                newmat._total_energy = site['material'].total_energy
                _ = newmat.cohesive_energy
                self.sites[i]['material'] = newmat
                newdef._total_energy = site['defect'].total_energy
                _ = newdef.cohesive_energy
                self.sites[i]['defect'] = newdef
        return self.sites

    def refresh(self):
        """Resets feature values."""
        self._features = None
        self._site_features = None
        self._normalized_site_features = None
        return None

    @property
    def site_features(self):
        """Computes site-specific features."""
        if self._site_features is None:
            print('Featurizing sites: ')
            for i, site in tqdm(enumerate(self.sites),
                                total=len(self.sites),
                                unit='sites'):
                dft = site['material'].get_features(site_index=site['site_index'],
                                                    atomic_symbol=self.use_atomic_symbols,
                                                    use_EMT=self.use_EMT)
                dft['material_image_index'] = site['material_image_index']
                dft['site_index'] = site['site_index']
                dft['DFT_site_stability'] = self.get_DFT_site_stability(site=site)
                if i == 0:
                    df = dft.copy(deep=True)
                else:
                    df = pd.concat([df, dft], axis=0, sort=False)
                    df = df.fillna(0)
                    df = df.reset_index(drop=True)
            self._site_features = df
        return self._site_features

    @property
    def normalized_site_features(self):
        """Computes normalized site-specific features.
        Numeric features are selected."""
        if self._normalized_site_features is None:
            x = self.site_features.select_dtypes(['number'])
            colnames = list(x.columns)
            x = x.values
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            df = pd.DataFrame(x_scaled)
            df.columns = colnames
            self._normalized_site_features = df
        return self._normalized_site_features

    def get_DFT_site_stability(self, site):
        """
        Computes site stability based on material,
        defect and reference dict.
        :param site: Materials, Featurizer site i.e. Materials object.
        :return: Site stability in eV.
        """
        e_mat = site['material'].total_energy
        atom = site['material'].atoms[site['site_index']].symbol
        e_atom = self.reference_dict[atom]
        e_def = site['defect'].total_energy
        try:
            e_site = e_mat - e_def - e_atom
        except:
            # print('Check; site may not be converged: \n')
            # print(site)
            # print('\n')
            e_site = np.NAN
        return e_site

    @property
    def features_of_all_sites(self):
        """Computes all features of all sites of a material.
        May take a while."""
        if self._features is None:
            for i, atoms in enumerate(self.images):
                mat = Material(atoms, reference_dict=self.reference_dict)
                for n in range(int(mat.natoms)):
                    dft = mat.get_features(site_index=n,
                                           atomic_symbol=self.use_atomic_symbols,
                                           use_EMT=self.use_EMT)
                    dft['material_number'] = i
                    if i == 0 and n == 0:
                        df = dft.copy(deep=True)
                    else:
                        df = pd.concat([df, dft], axis=0)
                        df = df.fillna(0)
                        df = df.reset_index(drop=True)
            collist = list(df.columns)[:-1]
            collist.insert(0, 'material_number')
            print(collist)
            df = df[collist]
            self._features = df
        return self._features

    @property
    def unique_sites(self):
        """Determines all unique sites across list of materials
        in preparation for a random selection."""
        us = []
        for i, atoms in enumerate(self.images):
            mat = Material(atoms)
            for n in range(int(mat.natoms)):
                us.append([i, n])
        a = np.array(us)
        return a

    def select_specific_sites(self, site_selection):
        """
        Selects specific sites from the atomic images and creates the site instances.
        :param site_selection: np.ndarray of shape 2*n_sites.
        :return: sites
        """
        self.selected_sites = site_selection
        self.sites = []

        for t in self.selected_sites:
            RS = {}
            RS['material_image_index'] = t[0]
            RS['site_index'] = t[1]
            RS['material'] = Material(self.images[t[0]])
            RS['material_composition'] = composition_name(RS['material'].atoms)
            RS['defect'] = Material(RS['material'].create_defect(site_index=t[1]))
            RS['defect_composition'] = composition_name(RS['defect'].atoms)
            self.sites.append(RS)
        return self.sites

    def select_random_sites(self, n_sites=10, seed=42, exclude=None):
        """
        Select unique combination of sites across all Materials
        without duplicates.
        :param n_sites: Total number of sites to be selected.
        :param seed: Random seed for reproducibility.
        :param exclude: Exclude sites from previous run (give np.array).
        :return: np.array [Material index, site index]
        """
        self.sites = []
        np.random.seed(seed=seed)

        if exclude is None:
            new_candidates = self.unique_sites
        else:
            ex_set = set(map(tuple, exclude))
            all_set = set(map(tuple, self.unique_sites))
            new_candidates = list(all_set - ex_set)
            dt = np.dtype('int,int')
            new_candidates = np.array(new_candidates, dtype=dt)
            new_candidates = np.array([list(tup) for tup in list(new_candidates)])

        random_sites = list(np.random.choice(len(new_candidates), size=n_sites))

        self.selected_sites = new_candidates[random_sites]

        for t in self.selected_sites:
            RS = {}
            RS['material_image_index'] = t[0]
            RS['site_index'] = t[1]
            RS['material'] = Material(self.images[t[0]])
            RS['material_composition'] = composition_name(RS['material'].atoms)
            RS['defect'] = Material(RS['material'].create_defect(site_index=t[1]))
            RS['defect_composition'] = composition_name(RS['defect'].atoms)
            self.sites.append(RS)
        return self.sites

    def write_site_trajectories(self, folderpath=os.getcwd()):
        """
        Creates folder with trajectories of site defect structure
        and corresponding defect-free material.
        :param folderpath: Path to folder.
        :return: Writes out ase trajectory files.
        """
        fp = folderpath
        if not fp.endswith('/'):
            fp = fp + '/'
        for RS in self.sites:
            # Material
            matpath = fp + 'material' + str(RS['material_image_index'])
            if not os.path.isdir(matpath):
                os.makedirs(matpath+'/defect_free/', exist_ok=True)
                ase.io.write(matpath + '/defect_free/init.traj', RS['material'].atoms)

            # Defect
            defpath = matpath + '/site' + str(RS['site_index']) + '_' + RS['defect_composition'] + '/'
            os.makedirs(defpath, exist_ok=True)
            ase.io.write(defpath + 'init.traj', RS['defect'].atoms)
        return None

    def read_sites(self, folderpath, filename, extend=False):
        """
        Reads in sites from pickle.
        :param folderpath: Full path to folder.
        :param filename: Pickle file name.
        :param extend: Extend previous list of sites.
        :return: Full list of sites.
        """
        if not folderpath.endswith('/'):
            folderpath = folderpath + '/'
        filepath = folderpath + filename
        if extend and self.sites is not None:
            new_sites = pickle.load(open(filepath, "rb"))
            self.sites.extend(new_sites)
        else:
            self.sites = pickle.load(open(filepath, "rb"))
        return self.sites

    def traj_to_sites(self, material_filepath,
                      defect_filepath,
                      extend=True):
        """
        From a given pair of defect-free material and 
        defect structure, create list of sites.
        :param material_filepath: Full filepath to defect-free trajectory.
        :param defect_filepath: Full filepath to defect trajectory.
        :param extend: If sites exist in the SiteFeaturizer object, 
        whether to extend or not
        :return: List o
        """
        material = ase.io.read(material_filepath)
        defect = ase.io.read(defect_filepath)

        if self.sites is None:
            self.sites = []

        RS = {}
        RS['material_image_index'] = len(self.sites)
        RS['site_index'] = get_site_index(material, defect)
        RS['material'] = Material(material, reference_dict=self.reference_dict)
        RS['material_composition'] = str(RS['material'].atoms.symbols)
        RS['defect'] = Material(defect, reference_dict=self.reference_dict)
        RS['defect_composition'] = str(RS['defect'].atoms.symbols)

        if extend:
            self.sites.append(RS)
        else:
            self.sites = [RS]

        return self.sites

    def save_sites(self, folderpath=None, filename=None):
        """
        Save sites as pickle file.
        :param folderpath: Full path to out folder.
        :param filename: Out file name.
        :return: Writes out pickle file.
        """
        if folderpath is None:
            folderpath = os.getcwd() + '/'
        if not folderpath.endswith('/'):
            folderpath = folderpath + '/'
        if filename is None:
            filename = 'site_selection_' + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '.p'
        os.makedirs(folderpath, exist_ok=True)
        pickle.dump(self.sites, open(folderpath + filename, "wb"))
        return None

    def save_dataset(self, folderpath=None, filename=None, normalized=False):
        """
        Save data set with features and site stabilities.
        :param folderpath: Full path to out folder.
        :param filename: Out file name, should end with tsv.
        :param normalized: Whether features should be normalized (numeric features only.)
        :return:
        """
        if folderpath is None:
            folderpath = os.getcwd() + '/'
        if not folderpath.endswith('/'):
            folderpath = folderpath + '/'
        if filename is None:
            filename = 'site_selection_' + datetime.datetime.now().strftime("%Y%m%d_%H%M") + '.tsv'
        os.makedirs(folderpath, exist_ok=True)
        if not normalized:
            df = self.site_features
        else:
            df = self.normalized_site_features
        df.to_csv(folderpath + filename, sep='\t')
        return None

    def remove_outlier(self, column='DFT_site_stability', value=0):
        # Remove outlier
        df = self.site_features
        idx = df[df[column] == value].index[0]
        self.sites.pop(idx)
        self.refresh()
        return None

class GAFeatureSelection:
    def __init__(self,
                 X,
                 y,
                 n_features=2,
                 population_size=50,
                 offspring_size=10,
                 clf=None,
                 starting_population=None):

        self.X = X
        self.y = y
        self.feature_names = list(self.X.columns)
        self.total_features = len(self.X.columns)
        if clf is None:
            self.clf = RandomForestRegressor(max_depth=5,
                                        random_state=42,
                                        n_estimators=50)
        else:
            self.clf = clf

        self.n_features = n_features
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.offspring_genes = None

        self.offspring = None
        self.selected_features = None

        if starting_population is None:
                self.genes = self.random_genes()
                self.population = self.get_fitness(genes=self.genes)
        else:
            self.population = starting_population

        self.genes = [tup[0] for tup in self.population]
        self.crossover()

        self.evolution = None

    @property
    def mean_population_fitness(self):
        m = np.mean([tup[1] for tup in self.population])
        return m

    @property
    def max_population_fitness(self):
        m = np.max([tup[1] for tup in self.population])
        return m

    def random_genes(self):
        # Create random starting population
        print('Initializing random population.')
        init_chromosome = [1] * self.n_features + (self.total_features - self.n_features) * [0]
        new_pop = []
        for _ in range(self.population_size):
            new_pop.append(random.sample(init_chromosome, len(init_chromosome)))
        self.genes = [[i for i, _ in enumerate(x) if _ == 1] for x in new_pop]
        return self.genes

    def _get_cv_score(self, X_gene):
        np.random.seed(42)
        scores = cross_val_score(self.clf, X_gene, self.y, cv=3)
        return np.mean(scores)

    def get_fitness(self, genes):
        results = []
        for gene in genes:
            X_gene = self.X[self.X.columns[gene]]
            gene_score = self._get_cv_score(X_gene=X_gene)
            results.append([gene, gene_score])
        return sorted(results, reverse=True, key=lambda tup: tup[1])

    def crossover(self):
        parent_genes = [tup[0] for tup in self.population]
        crossover_point = int(len(parent_genes[0]) / 2)
        offspring_genes = []
        for i in range(self.offspring_size):
            parent1 = parent_genes[i]
            random.shuffle(parent1)
            parent2 = parent_genes[i+i]
            random.shuffle(parent2)
            offspring_genes.append(parent1[:crossover_point] + parent2[crossover_point:])

        self.offspring_genes = self.mutate_duplicate_chromosomes(genes=offspring_genes)

        # Get offspring fitness
        self.offspring = self.get_fitness(genes=self.offspring_genes)

        # Lower parent fitness for gene diversity
        parents_size = len(self.offspring_genes) + 1
        probability_adjustment = self.mean_population_fitness
        for i in range(parents_size):
            self.population[i][1] = probability_adjustment

        # Update population
        self.population = self.population[:len(self.population) - len(self.offspring)]
        self.population.extend(self.offspring)
        self.population = sorted(self.population, reverse=True, key=lambda tup: tup[1])
        self.genes = [tup[0] for tup in self.population]
        return self.population

    def mutate_duplicate_chromosomes(self, genes):
        for i, gene in enumerate(genes):
            if len(gene) != len(set(gene)):
                n_duplicates = len(gene) - len(list(set(gene)))
                new_candidates = list(set(range(self.total_features)) - set(gene))

                base = list(set(gene))
                for _ in range(n_duplicates):
                    base.extend(random.sample(new_candidates, 1))
                    genes[i] = base
        return genes

    @property
    def homogeneity(self):
        gene_ensemble = set(list(np.array(self.genes).flatten()))
        return 1-(len(gene_ensemble)/self.total_features)

    def mutate(self):
        random_mutation_percentage = random.randint(5, 20)/100
        n_mutations = int(len(self.genes)*random_mutation_percentage)
        genes_copy = self.genes.copy()
        for i in range(n_mutations):
            random_gene_position = random.sample(range(self.population_size),1)[0]
            random_gene = genes_copy[random_gene_position]
            random_chromosome_position = random.sample(range(self.n_features), 1)[0]

            new_chromosome_candidates = list(set(range(self.total_features)) - set(random_gene))
            random_gene[random_chromosome_position] = random.sample(new_chromosome_candidates, 1)[0]
            genes_copy[random_gene_position] = random_gene
        self.genes = genes_copy
        return self.genes

    def evolve(self, generations=20):
        evolution = np.ndarray((generations, 4))
        for i in range(generations):
            print('Generation: '+str(i+1))
            self.crossover()
            evolution[i][0] = i
            evolution[i][1] = self.max_population_fitness
            evolution[i][2] = self.mean_population_fitness
            evolution[i][3] = self.homogeneity
            if self.homogeneity > 0.2:
                self.mutate()
            print('Max fitness: %5.2f ' % self.max_population_fitness)

        df_ev = pd.DataFrame(evolution)
        df_ev.columns = ['generation', 'max_population_fitness', 'mean_population_fitness', 'homogeneity']
        self.evolution = df_ev
        return self.evolution

    def plot_features(self, show=True):
        all_genes = np.array(self.genes).flatten()
        gene_occurrences = np.bincount(all_genes)
        features = list(self.X.columns)
        feature_occurrences = sorted(list(zip(features, gene_occurrences)), key=lambda tup: tup[1], reverse=True)
        occ = np.array(list(zip(*feature_occurrences))[1])
        occ = occ/occ.sum()*100
        feature_names = list(list(zip(*feature_occurrences))[0])

        fig, ax = plt.subplots(figsize=(8, 8))
        ax = sns.barplot(x=occ, y=feature_names)
        ax.set_xlabel('Occurence probability (%)')
        plt.tight_layout()

        if show:
            plt.show()
        return fig

    def plot_evolution_stats(self, show=True):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.evolution.generation, self.evolution.max_population_fitness, label='5fold cv - R2 max')
        ax.plot(self.evolution.generation, self.evolution.mean_population_fitness, label='5fold cv - R2 mean')
        ax.plot(self.evolution.generation, self.evolution.homogeneity, label='homogeneity')
        ax.legend(loc=2, bbox_to_anchor=(1.0, 1.0))
        ax.set_xlabel('Generation')
        plt.tight_layout()

        if show:
            plt.show()
        return fig

