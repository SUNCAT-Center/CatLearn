import ase.io
import catlearn
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

from catlearn.featurize import neighbor_matrix
from catlearn.featurize import *
from catlearn.featurize.slab_utilities import *

from itertools import chain, combinations
from matplotlib.lines import Line2D
from mendeleev import element

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from tqdm import tqdm

# pd.set_option('display.max_columns', 10)

# Plotting settings
plt.rc('text', usetex=False)
font = {'size': 28}
plt.rc('font', **font)

sns.set_style('white')
sns.set_palette(sns.hls_palette(8, h=0.5, l=0.4, s=0.5))

infile = ''

atom_dict = {'Ni': -1380.8341027932559,
             'Au': -1174.461466369728,
             'Pd': -1083.667223896331,
             'Pt': -998.5654412676529,
             'Ag': -1306.3829300840296,
             'Cu': -1611.535995452583}


def composition_name(atoms):
    """
    Takes an atoms object and returns stoichiometric composition names without duplicate elements.
    :param atoms: Atoms object.
    :return: str composition name.
    """
    itm = list(stoichiometry(atoms).items())
    itm = [str(e) for l in itm for e in l]
    return ''.join(itm)


def traj_to_reference_dict(folderpath):
    """
    Takes folder path with trajectories of reference atoms
    and molecules and transforms it to an energy dictionary.
    :param folderpath: Path to trajectories, which should be supplied in .traj or .json.
    :return: dict: refDict Reference dictionary.
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
    Read in file name including path to tsv, returns dataframe without index column.
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
    :return: site index integer
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
    """
    Find unique sets of n features.
    :param iterable: list of full possibilities.
    :param feature_dim: Combination size.
    :return: iterable chain with all unique (no duplicate) combinations.
    """
    s = list(iterable)  # allows duplicate elements
    ch = chain.from_iterable(combinations(s, r) for r in range(feature_dim, feature_dim + 1))
    sn = [x for x in ch]
    return sn



class Material:
    """Material object. Stores and computes atomic system and site properties."""
    def __init__(self, atoms, reference_dict='default'):
        """
        Instantiates Material object from atoms object.
        :param atoms: ASE atoms object
        :param reference_dict: Dictionary with atom energies in vacuum..
        """
        self.atoms = atoms
        if reference_dict == 'default':
            reference_dict = atom_dict
        self.reference_dict = reference_dict
        self.atomic_number_list = self.atoms.numbers
        self.atomic_numbers = list(set(self.atomic_number_list))
        self.cm = neighbor_matrix.connection_matrix(atoms, dx=2.0)
        self.cn_list = np.sum(self.cm, axis=1)
        self.composition = stoichiometry(self.atoms)
        self.composition_coefficients = list(self.composition.values())
        self.natoms = float(len(self.atoms.numbers))
        self.symbols = list(self.composition.keys())
        self.selected_sites = None
        self._total_energy = None
        self._cohesive_energy = None
        self.EMT_site_stability = None

        self.site_index = None
        self.site_symbol = None
        self.site_atomic_number = None
        self.site_e_val = None
        self.site_cn = None

        self.max_neighbor_cn = None
        self.min_neighbor_cn = None
        self.mean_neighbor_cn = None
        self.std_neighbor_cn = None

        self.site_cm = None
        self.max_neighbor_atomic_number = None
        self.min_neighbor_atomic_number = None
        self.mean_neighbor_atomic_number = None
        self.std_neighbor_atomic_number = None

        self.site_features = None

        self._site_angles_min = None
        self._site_angles_max = None
        self._site_angles_mean = None
        self._site_angles_std = None
        self._site_angles = None

        self._site_distances = None
        self._site_distance_mean = None
        self._site_distance_std = None
        self._site_distance_min = None
        self._site_distance_max = None

        self._system_features = None

    @property
    def total_energy(self):
        """
        Gets the total energy from atoms object.
        Make sure the trajectory or atoms object has energy.
        :return: float of energy in eV.
        """
        if self._total_energy is None:
            try:
                self._total_energy = float(self.atoms.get_potential_energy())  # eV
            except:
                self._total_energy = None
        return self._total_energy

    @property
    def cohesive_energy(self):
        """Computes the cohesive energy of the full system."""
        if self._cohesive_energy is None:
            c = self.total_energy
            for key, value in self.composition.items():
                atomic_energy = self.reference_dict[key]
                c = c - (atomic_energy * value)
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
        self.mean_neighbor_cn = neighbor_cn.mean()
        self.std_neighbor_cn = neighbor_cn.std()

        # Neighbor atomic number
        self.site_cm = self.cm[site_index]
        neighbor_atomic_numbers = self.atomic_number_list * self.cm[site_index]
        neighbor_atomic_numbers = np.ma.masked_equal(neighbor_atomic_numbers, 0)
        self.max_neighbor_atomic_number = neighbor_atomic_numbers.max()
        self.min_neighbor_atomic_number = neighbor_atomic_numbers.min()
        self.mean_neighbor_atomic_number = neighbor_atomic_numbers.mean()
        self.std_neighbor_atomic_number = neighbor_atomic_numbers.std()

        # EMT site stability
        if use_EMT:
            _ = self.get_EMT_site_stability(site_index=self.site_index)

        # Site atoms triad angles and pair distances
        _ = self.site_angles
        _ = self.site_distances

        # Gather all into feature list
        column_names = ['site_index', 'site_atomic_number', 'site_e_val', 'site_cn',
                        'min_neighbor_cn', 'max_neighbor_cn', 'mean_neighbor_cn', 'std_neighbor_cn',
                        'min_neighbor_atomic_number', 'max_neighbor_atomic_number',
                        'mean_neighbor_atomic_number', 'std_neighbor_atomic_number',
                        'site_distance_min', 'site_distance_max', 'site_distance_mean', 'site_distance_std',
                        'site_angles_min', 'site_angles_max', 'site_angles_mean', 'site_angles_std']
        if atomic_symbol:
            column_names = ['symbol'] + column_names
        if use_EMT:
            column_names = ['EMT_site_stability'] + column_names

        site_features = [self.site_index, self.site_atomic_number, self.site_e_val, self.site_cn,
                         self.min_neighbor_cn, self.max_neighbor_cn, self.mean_neighbor_cn, self.std_neighbor_cn,
                         self.min_neighbor_atomic_number, self.max_neighbor_atomic_number,
                         self.mean_neighbor_atomic_number, self.std_neighbor_atomic_number,
                         self._site_distance_min, self._site_distance_max,
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

    @property
    def site_angles(self):
        """Computes angles of unique triples of atoms with the site being the middle atom.
        Computes statistics from angles.
        :return: List of features."""
        cm_site = self.cm[self.site_index]
        cm_site[self.site_index] = 0
        neighbors = [i for i, x in enumerate(cm_site) if x == 1]
        pairs = unique_set(iterable=neighbors)

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

        # Unfortunately some angles are faulty
        angles = [x for x in angles if not np.isnan(x)]

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
        self._site_distance_min = np.min(self._site_distances)
        self._site_distance_max = np.max(self._site_distances)
        return self._site_distances

    @property
    def system_features(self):
        """Computes system-specific features.
        :return: pd.DataFrame with features."""
        df = pd.DataFrame()

        # general
        df['natoms'] = [self.natoms]
        df['DFT_cohesive_energy'] = [self.cohesive_energy]

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
        df['Z_mean'] = [np.mean(ans)]
        df['e_val_mean'] = [np.mean(evals)]

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
        Returns EMT site stability of site (only for selected metals).
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


class SiteFeaturizer:
    """Class to handle data set of sites, including atoms,
    reference energies, features and site generation, outlier removal."""
    def __init__(self,
                 images=None,
                 sites=None,
                 use_atomic_symbols=False,
                 use_EMT=False,
                 reference_dict='default'):
        """
        Instantiates the SiteFeaturizer class with:
        :param images: List of Atom objects, optional.
        :param sites: List of sites, such as those created by the class.
        :param use_atomic_symbols: Whether to use atomic symbols as features.
        :param use_EMT: Whether to use EMT site stabilities as features (some metals only).
        :param reference_dict: Dictionary of reference energies (atoms or molecule energies).
        """
        self.images = images
        self.sites = sites
        self.use_atomic_symbols = use_atomic_symbols
        self.use_EMT = use_EMT
        if reference_dict == 'default':
            reference_dict = atom_dict
        self.reference_dict = reference_dict

        self._all_possible_sites_features = None
        self._site_features = None
        self._normalized_site_features = None

    def _update_materials(self):
        """Re-instantiates sites from material and defect Atom objects
        and their total energies.
        Mainly for development purposes, e.g. when Material objects changed."""
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
        self._all_possible_sites_features = None
        self._site_features = None
        self._normalized_site_features = None
        return None

    @property
    def site_features(self):
        """Computes system and site-specific features for all sites."""
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
                    dfnew = dft.copy(deep=True)
                else:
                    dfnew = pd.concat([dfnew, dft], axis=0, sort=False)
                    dfnew = dfnew.fillna(0)
                    dfnew = dfnew.reset_index(drop=True)

            atom1_list = [x for x in list(dfnew.columns) if '1' in x]
            for atom1 in atom1_list:
                atom1_substitute = atom1.replace('1', '0')
                dfnew[atom1] = dfnew.apply(lambda row: row[atom1_substitute] if row[atom1] == 0 else row[atom1], axis=1)

            self._site_features = dfnew
        return self._site_features

    @property
    def normalized_site_features(self):
        """Computes normalized system- and site-specific features for all sites.
        Only numeric features are selected."""
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
        :param site: SiteFeaturizer site.
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
        """Computes all features of all atomic sites of a material.
        May take a while."""
        if self._all_possible_sites_features is None:
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
            self._all_possible_sites_features = df
        return self._all_possible_sites_features

    @property
    def unique_sites(self):
        """Determines all unique sites across list of materials
        in preparation for a random selection."""
        us = []
        for i, atoms in enumerate(self.images):
            mat = Material(atoms)
            # Fast enough for this case:
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

    def write_site_trajectories(self, folderpath='default'):
        """
        Creates folder with trajectories of site defect structure
        and corresponding defect-free material.
        :param folderpath: Path to folder.
        :return: Writes out ase trajectory files.
        """
        fp = folderpath
        if fp == 'default':
            fp = os.getcwd()
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
        if not os.path.isfile(material_filepath) or not os.path.isfile(defect_filepath):
            return self.sites

        material = ase.io.read(material_filepath)
        try:
            material.set_constraint()
        except:
            print('Constraint problems')
        defect = ase.io.read(defect_filepath)

        try:
            material.set_constraint()
        except:
            print('Constraint problems')

        if self.sites is None:
            self.sites = []

        RS = dict()
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

    def save_dataset(self,
                     folderpath=None,
                     filename=None,
                     normalized=False):
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
        """
        Removes outliers in specified column.
        :param column: Column in which to search for the value.
        :param value: Value considered the outlier.
        :return: None
        """
        df_tmp = self.site_features
        outlier_df = df_tmp[df_tmp[column] == value]
        if outlier_df.shape[0] > 0:
            index_list = outlier_df.index.values.tolist()
            df_tmp = df_tmp.drop(index_list, axis=0)
            self._site_features = df_tmp.reset_index(drop=True)
            for idx in index_list:
                self.sites.pop(idx)
        return None


class GAFeatureSelection:
    """Feature combination (model) selection using a genetic algorithm (GA)"""
    def __init__(self,
                 X,
                 y,
                 clf=None,
                 n_features=2,
                 population_size=100,
                 offspring_size=10,
                 scoring='cv',
                 n_cv=4,
                 random_state=42,
                 starting_population=None,
                 verbose=0):
        """
        Instantiates GAFeatureSelection object.
        :param X: feature data set of training data.
        :param y: targets of training data set.
        :param clf: classifier, or regressior (e.g. sklearn regression object)
        :param n_features: Number of features that the resulting model should have.
        :param population_size: How many feature combinations to start with (the more the better).
        :param offspring_size: How many offsprings should be created by mating the best candidates.
        :param scoring: How to determine the fitness of the model, cv uses cross validation R2.
        :param n_cv: validation set split multiple.
        :param random_state: random state integer for reproducibility.
        :param starting_population: GAFeatureSelection object can start from
        a pre-converged/pre-computed population.
        :param verbose: Whether to allow printing output.
        """

        self.verbose = verbose
        self.random_state = random_state
        self.X = X
        self.y = y
        self.scoring = scoring
        self.n_cv = n_cv
        self.feature_names = list(self.X.columns)
        self.total_features = len(self.X.columns)
        if clf is None:
            self.clf = RandomForestRegressor(max_depth=5,
                                             random_state=42,
                                             n_estimators=50)
            print('Using RandomForestRegressor.')
        else:
            self.clf = clf

        self.n_features = n_features
        self.all_feature_combinations = None

        self.population_size = population_size
        self.offspring_size = offspring_size
        self.offspring_chromosomes = None

        self.offspring = None
        self.selected_features = None

        if starting_population is None:
            print('Creating random population.')
            self.genes = self.random_genes(n=self.population_size)
            print('Calculating population metrics.')
            self.population = self.get_fitness(chromosomes=self.genes)
        else:
            self.population = starting_population

        self.genes = [tup[0] for tup in self.population]
        self._feature_frequencies = self.feature_frequencies
        self.feature_frequency_evolution = {}

        self.evolution = None
        self.evolution_history = None

    @property
    def mean_population_fitness(self):
        """Computes average of population fitness metrics."""
        m = np.mean([tup[2] for tup in self.population])
        return m

    def population_percentile(self, percentile=75):
        """Computes percentile of population fitness metrics."""
        m = np.percentile([tup[2] for tup in self.population], percentile)
        return m

    @property
    def max_population_fitness(self):
        """Computes max of population fitness metrics."""
        m = np.max([tup[2] for tup in self.population])
        return m

    def random_genes(self, n=1):
        """
        From the unique sets of n_features combinations, returns the first n.
        :param n: Number of chromosomes to return.
        :return: List of unique random chromosomes of length n.
        """
        random.seed(self.random_state)
        if self.all_feature_combinations is None:
            unique_combinations = []
            for feature_combination in unique_set(list(range(self.total_features)), feature_dim=self.n_features):
                unique_combinations.append(list(feature_combination))
            random.shuffle(unique_combinations)
            self.all_feature_combinations = unique_combinations
        selection = self.all_feature_combinations.copy()
        random.shuffle(selection)
        return selection[:n]

    def get_chromosome_score(self, X_chromosome):
        """
        Computes fitness using the subset of data in X_chromosome.
        :param X_chromosome: subset of full data set, containing only a selection of the features.
        :return: mean R2 or keras history last column entry.
        """
        np.random.seed(self.random_state)
        # Use either cross validation
        if self.scoring == 'cv':
            scores = cross_val_score(self.clf, X_chromosome, np.array(self.y), cv=self.n_cv)
            return np.mean(scores)
        # Or keras history in the case of neural networks (based on keras/tensorflow)
        else:
            try:
                history = self.clf.fit(X_chromosome, np.array(self.y))
                return history.history[self.scoring][-1]
            except:
                raise ValueError('Use either "cv" or keras history metrics.')

    def get_fitness(self, chromosomes):
        """
        Compute the fitness (derived from the performance metric, but not equal to it,
        as it is updated in the cross over process.
        :param chromosomes: List of chromosomes, which in turn are list of features.
        :return: List of the chromosome, and the performance metric (twice),
        which is necessary to keep track of performance during cross over.
        """
        results = []
        for n_chromosome, chromosome in enumerate(chromosomes):
            if self.verbose:
                print('Evaluating chromosome '+str(n_chromosome+1)+'/'+str(len(chromosomes)))

            X_chromosome = self.X[self.X.columns[chromosome]]
            chromosome_score = self.get_chromosome_score(X_chromosome=X_chromosome)

            if self.verbose:
                print('Chromosome score: '+str(chromosome_score))

            results.append([chromosome, chromosome_score, chromosome_score])
        return sorted(results, reverse=True, key=lambda tup: tup[1])

    def crossover(self):
        """Parent gene mating. Out of the best m+1 parents, a pairwise crossover is generated as the offspring,
        with m being the offspring size. """
        parent_genes = [tup[0] for tup in self.population]
        crossover_point = int(len(parent_genes[0]) / 2)
        offspring_chromosomes = []
        for i in range(self.offspring_size):
            parent1 = parent_genes[i]
            random.shuffle(parent1)
            parent2 = parent_genes[i+i]
            random.shuffle(parent2)
            offspring_chromosomes.append(parent1[:crossover_point] + parent2[crossover_point:])

        self.offspring_chromosomes = self.mutate_duplicate_chromosomes(chromosome_list=offspring_chromosomes)

        # Get offspring fitness
        self.offspring = self.get_fitness(chromosomes=self.offspring_chromosomes)

        # Lower parent fitness for gene diversity
        parents_size = len(self.offspring_chromosomes) + 1
        probability_adjustment = self.population_percentile()
        for i in range(parents_size):
            self.population[i][1] = self.population[i][1]*probability_adjustment

        # Update population
        self.population = self.population[:len(self.population) - len(self.offspring)]
        self.population.extend(self.offspring)
        self.population = sorted(self.population, reverse=True, key=lambda tup: tup[1])
        self.genes = [tup[0] for tup in self.population]
        return self.population

    def mutate_duplicate_chromosomes(self, chromosome_list):
        """
        During cross-over duplicate genes can occur.
        These will be replaced by random genes, while avoiding duplicate generation.
        :param chromosome_list: list of chromosomes to check.
        :return: duplicate-free chromosome list.
        """
        for i, chromosome in enumerate(chromosome_list):
            if len(chromosome) != len(set(chromosome)):
                n_duplicates = len(chromosome) - len(list(set(chromosome)))
                new_candidates = list(set(range(self.total_features)) - set(chromosome))

                chromosome = list(set(chromosome))
                chromosome.extend(np.random.choice(new_candidates, size=n_duplicates, replace=False))
                chromosome_list[i] = chromosome
        return chromosome_list

    @property
    def homogeneity(self):
        """Measure of whether all possible features are still in the population."""
        gene_ensemble = set(list(np.array(self.genes).flatten()))
        return 1-(len(gene_ensemble)/self.total_features)

    def mutate(self):
        """Increase stochastic effects to enhance optimization."""
        random_mutation_percentage = random.randint(1, 20)/100
        n_mutations = int(len(self.genes)*random_mutation_percentage)
        genes_copy = self.genes.copy()
        for i in range(n_mutations):
            random_gene_position = random.sample(range(self.population_size), 1)[0]
            random_gene = genes_copy[random_gene_position]
            random_chromosome_position = random.sample(range(self.n_features), 1)[0]

            new_chromosome_candidates = list(set(range(self.total_features)) - set(random_gene))
            random_gene[random_chromosome_position] = random.sample(new_chromosome_candidates, 1)[0]
            genes_copy[random_gene_position] = random_gene
        self.genes = genes_copy
        return self.genes

    def evolve(self, generations=20):
        """
        Run genetic algorithm for generations number of generations.
        :param generations: number of generations
        :return: evolution history with evolution of performance metrics.
        """
        evolution = np.ndarray((generations, 4))
        for i in range(generations):
            print('Generation: ' + str(i))
            # start with stats from random population
            self.feature_frequency_evolution[len(self.feature_frequency_evolution.keys())] = self.feature_frequencies
            evolution[i][0] = i
            evolution[i][1] = self.max_population_fitness
            evolution[i][2] = self.mean_population_fitness
            evolution[i][3] = self.homogeneity

            self.crossover()
            self.mutate()

            print('Max fitness: %5.2f ' % self.max_population_fitness)
            print('Mean fitness: %5.2f ' % self.mean_population_fitness)

        df_ev = pd.DataFrame(evolution)
        df_ev.columns = ['generation', 'max_population_fitness', 'mean_population_fitness', 'homogeneity']

        self.evolution = df_ev
        if self.evolution_history is None:
            self.evolution_history = self.evolution
        else:
            self.evolution_history = pd.concat([self.evolution_history, self.evolution], axis=0).reset_index(drop=True)
            self.evolution_history.generation = self.evolution_history.index
        return self.evolution

    @property
    def feature_frequencies(self):
        """Compute how often every feature occurs in the population."""
        all_genes = np.array(self.genes).flatten()
        gene_occurrences = np.bincount(all_genes)
        features = list(self.X.columns)
        feature_occurrences = sorted(list(zip(features, gene_occurrences)), key=lambda tup: tup[1], reverse=True)
        occ = np.array(list(zip(*feature_occurrences))[1])
        self._feature_frequencies = dict(zip(list(list(zip(*feature_occurrences))[0]), occ/occ.sum()*100))
        return self._feature_frequencies

    def plot_features(self,
                      labels=None,
                      frequencies=None,
                      show=True):
        """Plots feature frequencies"""
        print('Plotting feature frequencies.')
        if labels is None:
            labels = list(self.feature_frequencies.keys())
        if frequencies is None:
            frequencies = list(self.feature_frequencies.values())
        fig, ax = plt.subplots(figsize=(12, int(len(labels)*0.6)))
        ax = sns.barplot(x=frequencies, y=labels)
        ax.set_xlabel(r'Occurrence probability (\%)')
        plt.tight_layout()

        if show:
            plt.show()

        return fig, ax

    def plot_evolution_stats(self,
                             full_history=True,
                             show=True):
        """Plots evolution history."""
        print('Plotting evolution statistics.')
        if full_history:
            df_stats = self.evolution_history
        else:
            df_stats = self.evolution
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(df_stats.generation, df_stats.max_population_fitness, lw=3, ls='-', label=r'$\mathrm{R^{2}}$-cv max')
        ax.plot(df_stats.generation, df_stats.mean_population_fitness, lw=3, ls='--', label=r'$\mathrm{R^{2}}$-cv mean')
        ax.plot(df_stats.generation, df_stats.homogeneity, lw=3, ls='-.', label='Homogeneity')
        # ax.legend(loc=2, bbox_to_anchor=(1.0, 1.0))
        ax.legend()
        ax.set_title('Evolution statistics')
        ax.set_xlabel('Generation')
        plt.tight_layout()

        if show:
            plt.show()
        return fig, ax
