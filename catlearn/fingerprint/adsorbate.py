"""Slab adsorbate fingerprint functions for machine learning."""
import numpy as np

from ase.symbols import string2symbols
from ase.data import ground_state_magnetic_moments as gs_magmom
from ase.data import atomic_numbers, chemical_symbols

from catlearn.featurize.periodic_table_data import (list_mendeleev_params,
                                                    default_params, get_radius,
                                                    electronegativities,
                                                    make_labels)
from catlearn.featurize.base import BaseGenerator, check_labels


default_adsorbate_fingerprinters = ['mean_chemisorbed_atoms',
                                    'mean_site',
                                    'max_site',
                                    'min_site',
                                    'median_site',
                                    'sum_site',
                                    'mean_surf_ligands',
                                    'term',
                                    'bulk',
                                    'strain',
                                    'en_difference_ads',
                                    'en_difference_chemi',
                                    'en_difference_active',
                                    'count_chemisorbed_fragment',
                                    'generalized_cn',
                                    'bag_cn',
                                    'bag_atoms_ads',
                                    'bag_edges_ads',
                                    'bag_edges_chemi']

extra_slab_params = ['atomic_radius',
                     'heat_of_formation',
                     'dft_bulk_modulus',
                     'dft_density',
                     'dbcenter',
                     'dbfilling',
                     'dbwidth',
                     'dbskew',
                     'dbkurt',
                     'oxistates',
                     'block',
                     'econf',
                     'ionenergies']


# Text based feature.
facetdict = {'001': [1.], '0001step': [2.], '100': [3.],
             '110': [4.], '111': [5.], '211': [6.], '311': [7.],
             '532': [8.]}


class AdsorbateFingerprintGenerator(BaseGenerator):
    def __init__(self, **kwargs):
        """Class containing functions for fingerprint generation.

        Parameters
        ----------
        params : list
            An optional list of parameters upon which to generate features.
        cn_max : int
            Optional integer for maximum expected coordination number.
        ion_number : int
            Optional atomic number.
        ion_charge : int
            Optional formal charge of that element.
        """
        # Slab periodic table parameters.
        if not hasattr(self, 'params'):
            self.slab_params = kwargs.get('params')

        if self.slab_params is None:
            self.slab_params = default_params + extra_slab_params

        super(AdsorbateFingerprintGenerator, self).__init__(**kwargs)

    def term(self, atoms=None):
        """Return a fingerprint vector with propeties averaged over the
        termination atoms.

        Parameters
        ----------
        atoms : object
        """
        labels = make_labels(self.slab_params, '', '_term')
        labels.append('ground_state_magmom_term')
        if atoms is None:
            return labels
        else:
            if ('key_value_pairs' in atoms.info and
                    'term' in atoms.info['key_value_pairs']):
                term = atoms.info['key_value_pairs']['term']
                numbers = [atomic_numbers[s] for s in string2symbols(term)]
            elif 'termination_atoms' in atoms.subsets:
                term = atoms.subsets['termination_atoms']
                numbers = atoms.numbers[term]
            else:
                raise NotImplementedError("termination fingerprint.")
            dat = list_mendeleev_params(numbers, params=self.slab_params)
            result = list(np.nanmean(dat, axis=0))
            result += [np.nanmean([gs_magmom[z] for z in numbers])]
            check_labels(labels, result, atoms)
            return result

    def bulk(self, atoms=None):
        """Return a fingerprint vector with propeties averaged over
        the bulk atoms.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        labels = make_labels(self.slab_params, '', '_bulk')
        labels.append('ground_state_magmom_bulk')
        if atoms is None:
            return labels
        else:
            if ('key_value_pairs' in atoms.info and
                    'bulk' in atoms.info['key_value_pairs']):
                bulk = atoms.info['key_value_pairs']['bulk']
                numbers = [atomic_numbers[s] for s in string2symbols(bulk)]
            elif 'bulk_atoms' in atoms.subsets:
                bulk = atoms.subsets['bulk_atoms']
                numbers = atoms.numbers[bulk]
            else:
                raise NotImplementedError("bulk fingerprint.")
            dat = list_mendeleev_params(numbers, params=self.slab_params)
            result = list(np.nanmean(dat, axis=0))
            result += [np.nanmean([gs_magmom[z] for z in numbers])]
            check_labels(labels, result, atoms)
            return result

    def mean_chemisorbed_atoms(self, atoms=None):
        """Function that takes an atoms objects and returns a fingerprint
        vector containing properties of the closest add atom to a surface
        metal atom.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        extra_ads_params = ['atomic_radius', 'heat_of_formation',
                            'oxistates', 'block', 'econf', 'ionenergies']
        labels = make_labels(default_params + extra_ads_params, '', '_ads1')
        labels.append('ground_state_magmom_ads1')
        if atoms is None:
            return labels
        else:
            # Get atomic number of alpha adsorbate atom.
            chemisorbed_atoms = atoms.subsets['chemisorbed_atoms']
            numbers = atoms.numbers[chemisorbed_atoms]
            # Import CatLearn data on that element.
            dat = list_mendeleev_params(numbers, params=default_params +
                                        extra_ads_params)
            result = list(np.nanmean(dat, axis=0))
            result += [np.nanmean([gs_magmom[z] for z in numbers])]
            check_labels(labels, result, atoms)
            return result

    def mean_site(self, atoms=None):
        """Function that takes an atoms objects and returns a fingerprint
        vector with properties averaged over the surface metal atoms
        closest to an add atom.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        labels = make_labels(self.slab_params, '', '_site_av')
        labels.append('ground_state_magmom_site_av')
        if atoms is None:
            return labels
        else:
            numbers = [atoms[j].number for j in atoms.subsets['site_atoms']]
            dat = list_mendeleev_params(numbers, params=self.slab_params)
            result = list(np.nanmean(dat, axis=0))
            result += [np.nanmean([gs_magmom[z] for z in numbers])]
            check_labels(labels, result, atoms)
            return result

    def min_site(self, atoms=None):
        """Function that takes an atoms objects and returns a fingerprint
        vector with properties averaged over the surface metal atoms
        closest to an add atom.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        labels = make_labels(self.slab_params, '', '_site_min')
        labels.append('ground_state_magmom_site_min')
        if atoms is None:
            return labels
        else:
            numbers = [atoms[j].number for j in atoms.subsets['site_atoms']]
            dat = list_mendeleev_params(numbers, params=self.slab_params)
            result = list(np.nanmin(dat, axis=0))
            result += [np.nanmin([gs_magmom[z] for z in numbers])]
            check_labels(labels, result, atoms)
            return result

    def max_site(self, atoms=None):
        """Function that takes an atoms objects and returns a fingerprint
        vector with properties averaged over the surface metal atoms
        closest to an add atom.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        labels = make_labels(self.slab_params, '', '_site_max')
        labels.append('ground_state_magmom_site_max')
        if atoms is None:
            return labels
        else:
            numbers = [atoms[j].number for j in atoms.subsets['site_atoms']]
            dat = list_mendeleev_params(numbers, params=self.slab_params)
            result = list(np.nanmax(dat, axis=0))
            result += [np.nanmax([gs_magmom[z] for z in numbers])]
            check_labels(labels, result, atoms)
            return result

    def median_site(self, atoms=None):
        """Function that takes an atoms objects and returns a fingerprint
        vector with properties averaged over the surface metal atoms
        closest to an add atom.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        labels = make_labels(self.slab_params, '', '_site_med')
        labels.append('ground_state_magmom_site_med')
        if atoms is None:
            return labels
        else:
            numbers = [atoms[j].number for j in atoms.subsets['site_atoms']]
            dat = list_mendeleev_params(numbers, params=self.slab_params)
            result = list(np.nanmedian(dat, axis=0))
            result += [np.nanmedian([gs_magmom[z] for z in numbers])]
            check_labels(labels, result, atoms)
            return result

    def sum_site(self, atoms=None):
        """Function that takes an atoms objects and returns a fingerprint
        vector with properties summed over the surface metal atoms
        closest to an add atom.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        labels = make_labels(self.slab_params, '', '_site_sum')
        labels.append('ground_state_magmom_site_sum')
        if atoms is None:
            return labels
        else:
            numbers = [atoms[j].number for j in atoms.subsets['site_atoms']]
            dat = list_mendeleev_params(numbers, params=self.slab_params)
            result = list(np.nansum(dat, axis=0))
            result += [np.nansum([gs_magmom[z] for z in numbers])]
            check_labels(labels, result, atoms)
            return result

    def generalized_cn(self, atoms):
        """Returns the averaged generalized coordination number over the site.
        Calle-Vallejo et al. Angew. Chem. Int. Ed. 2014, 53, 8316-8319.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        if atoms is None:
            return ['av_cn_site', 'cn_site', 'gcn_site', 'cn_ads1', 'gcn_ads1']
        site = atoms.subsets['site_atoms']

        if len(site) == 1:
            cn_max = 12
        elif len(site) == 2:
            cn_max = 18
        elif len(site) == 3:
            cn_max = 22
        elif len(site) == 4:
            cn_max = 26
        elif len(site) == 5:
            cn_max = 30
        elif len(site) >= 6:
            raise AssertionError('site of ' + str(len(site)) + ' atoms')

        slab = atoms.subsets['slab_atoms']
        cm = np.array(atoms.connectivity)
        np.fill_diagonal(cm, 0)

        site_neighbors = list(np.unique(atoms.subsets['ligand_atoms']))

        # Site coordination number.
        cn_site = len(site_neighbors)

        # Average coordination number of the site.
        av_cn_site = 0.
        for j, atom in enumerate(site):
            av_cn_site += np.sum(cm[atom, :][slab])
        av_cn_site /= len(site)

        # Generalized coordination number.
        gcn_site = 0.
        for j, btom in enumerate(site_neighbors):
            cn_nn = np.sum(cm[btom, :][slab])
            gcn_site += cn_nn
        gcn_site /= (cn_max * len(site_neighbors))

        # Average coordination number of chemisorbing atoms.
        chemi = atoms.subsets['chemisorbed_atoms']
        cn_chemi = 0.
        for atom in chemi:
            row = cm[atom, :]
            cn = len([k for i, k in enumerate(row) if k > 0])
            cn_chemi += cn
            gcn_chemi = 0.
            for j, btom in enumerate(row):
                if btom > 0:
                    cn = len([k for k in cm[btom, :] if k > 0])
                    gcn_chemi += btom * cn / 12

        return [av_cn_site, cn_site, gcn_site,
                cn_chemi / len(chemi), gcn_chemi / len(chemi)]

    def bag_cn(self, atoms):
        """Count the number of neighbors of the site, which has a n number of
        neighbors. This is equivalent to a bag of coordination numbers over the
        site neighbors.
        These can be used in the "alpha parameters" linear model.

        Please cite:
        Roling LT, Abild-Pedersen F.
        Structure-Sensitive Scaling Relations:
        Adsorption Energies from Surface Site Stability.
        ChemCatChem. 2018 Apr 9;10(7):1643-50.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        labels = ['count_' + str(j) + 'nn_site' for j in range(3, 31)]
        if atoms is None:
            return labels
        else:
            # Only count neighbors once.
            site_neighbors = list(np.unique(atoms.subsets['ligand_atoms']))
            slab = atoms.subsets['slab_atoms']

            fingerprint_nn = np.zeros(28)
            cm = np.array(atoms.connectivity)
            np.fill_diagonal(cm, 0)
            for i, atom in enumerate(site_neighbors):
                fingerprint_nn[np.sum(cm[atom, :][slab])] += 1

            fingerprint = list(fingerprint_nn)
            return fingerprint

    def bag_cn_general(self, atoms):
        """Count the number of neighbors of the site, which has a n number of
        neighbors. This is equivalent to a bag of coordination numbers over the
        site neighbors. These can be used in the "alpha parameters" linear
        model for alloys.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        # range of element types.
        raise NotImplementedError("In development.")

    def bag_atoms_ads(self, atoms=None):
        """Function that takes an atoms object and returns a fingerprint
        vector containing the count of each element in the
        adsorbate.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        if atoms is None:
            try:
                labels = ['bag_ads_' + chemical_symbols[z] for z in
                          self.atom_types]
            except TypeError as error:
                error.message += '\n Default atom_types require data to be' + \
                    ' generated before feature labels. Call return_vec first.'
                raise
            return labels
        else:
            bag = np.zeros(len(self.atom_types))
            for atom in atoms.subsets['ads_atoms']:
                bag[self.atom_types.index(atoms[atom].number)] += 1

            return bag

    def count_chemisorbed_fragment(self, atoms=None):
        """Function that takes an atoms objects and returns a fingerprint
        vector containing the count over atom types,
        that are neighbors to the chemisorbing atom.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        if atoms is None:
            labels = ['bag_chemi_nn_' + chemical_symbols[z] for z in
                      self.atom_types] + ['boc_site', 'n_site']
            return labels
        else:
            # Adsorbate atoms connected to the surface.
            chemi = atoms.subsets['chemisorbed_atoms']
            cm = np.array(atoms.connectivity)

            # Prepare bag of atom types.
            bag = np.zeros(len(self.atom_types))
            # Loop over elements that are present in the data set.
            for j, z in enumerate(self.atom_types):
                # Add neighbor counts to the bag of the relevant element.
                bag[j] += np.sum(cm[:, chemi] * np.vstack(atoms.numbers == z))

            boc_site = np.sum(cm[:, chemi][atoms.subsets['site_atoms'], :])
            site = len(atoms.subsets['site_atoms'])

            return list(bag) + [boc_site, site]

    def mean_surf_ligands(self, atoms=None):
        """Function that takes an atoms objects and returns a fingerprint
        vector containing the count of nearest neighbors and properties of
        the nearest neighbors.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        labels = ['nn_surf_ligands', 'identnn_surf_ligands']
        labels += make_labels(self.slab_params, '', '_surf_ligands')
        labels.append('ground_state_magmom_surf_ligands')
        if atoms is None:
            return labels
        else:
            ligand_atoms = atoms.subsets['ligand_atoms']
            numbers = atoms.numbers[ligand_atoms]
            # Import CatLearn data on that element.
            dat = list_mendeleev_params(numbers, params=self.slab_params)
            result = list(np.nanmean(dat, axis=0))
            result += [np.nanmean([gs_magmom[z] for z in numbers])]
            # Append count of ligand atoms.
            result = [len(ligand_atoms), len(np.unique(numbers))] + result
            check_labels(labels, result, atoms)
            return result

    def ads_sum(self, atoms=None):
        """Function that takes an atoms objects and returns a fingerprint
        vector with averages of the atomic properties of the adsorbate.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        ads_params = default_params + ['econf', 'ionenergies']
        labels = make_labels(ads_params, '', '_ads_sum')
        labels.append('ground_state_magmom_ads_sum')
        if atoms is None:
            return labels
        else:
            numbers = [atoms[j].number for j in atoms.subsets['ads_atoms']]
            dat = list_mendeleev_params(numbers, params=ads_params)
            result = list(np.nansum(dat, axis=0))
            result += [np.nansum([gs_magmom[z] for z in numbers])]
            check_labels(labels, result, atoms)
            return result

    def ads_av(self, atoms=None):
        """Function that takes an atoms objects and returns a fingerprint
        vector with averages of the atomic properties of the adsorbate.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        ads_params = default_params + ['econf', 'ionenergies']
        labels = make_labels(ads_params, '', '_ads_av')
        labels.append('ground_state_magmom_ads_av')
        if atoms is None:
            return labels
        else:
            numbers = [atoms[j].number for j in atoms.subsets['ads_atoms']]
            dat = list_mendeleev_params(numbers, params=ads_params)
            result = list(np.nanmean(dat, axis=0))
            result += [np.nanmean([gs_magmom[z] for z in numbers])]
            check_labels(labels, result, atoms)
            return result

    def strain(self, atoms=None):
        """Return a fingerprint with the espected strain of the
        site atoms and the termination atoms.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        if atoms is None:
            return ['strain_site', 'strain_term']
        else:
            if ('key_value_pairs' in atoms.info and
                    'term' in atoms.info['key_value_pairs']):
                term = atoms.info['key_value_pairs']['term']
                term_numbers = [atomic_numbers[s] for s in
                                string2symbols(term)]
            elif 'termination_atoms' in atoms.subsets:
                term = atoms.subsets['termination_atoms']
                term_numbers = atoms.numbers[term]
            else:
                raise NotImplementedError("strain fingerprint.")
            if ('key_value_pairs' in atoms.info and
                    'bulk' in atoms.info['key_value_pairs']):
                bulk = atoms.info['key_value_pairs']['bulk']
                bulk_numbers = [atomic_numbers[s] for s in
                                string2symbols(bulk)]
            elif 'bulk_atoms' in atoms.subsets:
                bulk = atoms.subsets['bulk_atoms']
                bulk_numbers = atoms.numbers[bulk]
            else:
                raise NotImplementedError("strain fingerprint.")
            site = atoms.subsets['site_atoms']
            site_numbers = atoms.numbers[site]
            rbulk = []
            rterm = []
            rsite = []
            for b in bulk_numbers:
                rbulk.append(get_radius(b))
            for t in term_numbers:
                rterm.append(get_radius(t))
            for z in site_numbers:
                rsite.append(get_radius(z))
            av_term = np.average(rterm)
            av_bulk = np.average(rbulk)
            av_site = np.average(rsite)
            strain_site = (av_site - av_bulk) / av_bulk
            strain_term = (av_term - av_bulk) / av_bulk
            return [strain_site, strain_term]

    def bag_edges_ads(self, atoms):
        """Returns bag of connections, counting only the bonds within the
        adsorbate.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        # range of element types
        n_elements = len(self.atom_types)
        symbols = np.array([chemical_symbols[z] for z in self.atom_types])
        rows, cols = np.meshgrid(symbols, symbols)
        pairs = np.core.defchararray.add(rows, cols)
        labels = ['boc_' + c + '_ads' for c in
                  pairs[np.triu_indices_from(pairs)]]
        if atoms is None:
            return labels
        else:
            # empty bag of connection types.
            boc = np.zeros([n_elements, n_elements])

            natoms = len(atoms)
            cm = np.array(atoms.connectivity)
            np.fill_diagonal(cm, 0)

            # Ignore connections with and within the slab.
            slab = atoms.subsets['slab_atoms']
            cm[slab, :] = 0
            cm[:, slab] = 0

            bonds = np.where(np.ravel(np.triu(cm)) > 0)[0]
            for b in bonds:
                # Get bonded atomic numbers.
                z_row, z_col = np.unravel_index(b, [natoms, natoms])
                bond_index = sorted((atoms.numbers[z_row],
                                     atoms.numbers[z_col]))
                bond_type = tuple((self.atom_types.index(bond_index[0]),
                                   self.atom_types.index(bond_index[1])))
                # Count bonds in upper triangle.
                boc[bond_type] += 1
            return list(boc[np.triu_indices_from(boc)])

    def bag_edges_chemi(self, atoms):
        """Returns bag of connections, counting only the bonds within the
        adsorbate and the connections between adsorbate and surface.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        # range of element types
        n_elements = len(self.atom_types)
        symbols = np.array([chemical_symbols[z] for z in self.atom_types])
        # Make array of connection types.
        rows, cols = np.meshgrid(symbols, symbols)
        pairs = np.core.defchararray.add(rows, cols)
        labels = ['boc_' + c + '_chemi' for c in
                  pairs[np.triu_indices_from(pairs)]]
        if atoms is None:
            return labels
        # empty bag of connection types.
        boc = np.zeros([n_elements, n_elements])

        natoms = len(atoms)
        cm = np.array(atoms.connectivity)
        np.fill_diagonal(cm, 0)

        # Ignore connections within the slab.
        slab = atoms.subsets['slab_atoms']
        cm[slab, :][:, slab] = 0
        cm[:, slab][slab, :] = 0

        bonds = np.where(np.ravel(np.triu(cm)) > 0)[0]
        for b in bonds:
            # Get bonded atomic numbers.
            z_row, z_col = np.unravel_index(b, [natoms, natoms])
            bond_index = sorted((atoms.numbers[z_row],
                                 atoms.numbers[z_col]))
            bond_type = tuple((self.atom_types.index(bond_index[0]),
                               self.atom_types.index(bond_index[1])))
            # Count bonds in upper triangle.
            boc[bond_type] += 1

        return list(boc[np.triu_indices_from(boc)])

    def bag_edges_all(self, atoms):
        """Returns bag of connections, counting all bonds within the
        adsorbate and between adsorbate atoms and surface. If we assign an
        energy to each type of bond, considering first neighbors only,
        this fingerprint would work independently in a linear model. The length
        of the vector is atom_types * ads_atom_types.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        # number of element types.
        n_elements = len(self.atom_types)
        n_elements_ads = len(self.ads_atom_types)

        # range of element types.
        symbols = np.array([chemical_symbols[z] for z in self.atom_types])
        ads_symbols = np.array([chemical_symbols[z] for z
                                in self.ads_atom_types])

        # Array of pairs.
        rows, cols = np.meshgrid(symbols, ads_symbols)

        # Add pairs to make labels.
        pairs = np.core.defchararray.add(rows, cols)
        labels = ['bea_' + c + '_ads' for c in
                  pairs[np.triu_indices_from(pairs)]]
        if atoms is None:
            return labels
        else:
            # empty bag of connection types.
            boc = np.zeros([n_elements_ads, n_elements])

            natoms = len(atoms)
            ads_atoms = atoms.subsets['ads_atoms']
            # n_ads_atoms = len(atoms.subsets['ads_atoms'])
            cm = np.array(atoms.connectivity)[ads_atoms, :]
            np.fill_diagonal(cm, 0)

            bonds = np.where(np.ravel(np.triu(cm)) > 0)[0]
            for b in bonds:
                # Get bonded atomic numbers.
                z_ads, z_all = np.unravel_index(b, [natoms, natoms])
                bond_index = (atoms.numbers[ads_atoms][z_ads],
                              atoms.numbers[z_all])
                bond_type = tuple((self.ads_atom_types.index(bond_index[0]),
                                   self.atom_types.index(bond_index[1])))
                # Count bonds in upper triangle.
                boc[bond_type] += 1
            return list(boc[np.triu_indices_from(boc)])

    def en_difference_ads(self, atoms=None):
        """Returns a list of electronegativity metrics, squared and summed over
        bonds within the adsorbate atoms.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        labels = ['dist_' + s + '_ads' for s in electronegativities]
        if atoms is None:
            return labels
        cm = atoms.connectivity
        ads = atoms.subsets['ads_atoms']
        bonds = cm[ads, :][:, ads]
        ads_numbers = atoms.numbers[ads]
        en_ads = list_mendeleev_params(ads_numbers, electronegativities)
        delta_en = (en_ads[:, np.newaxis, :] -
                    en_ads[np.newaxis, :, :]) ** 2
        en_result = list(np.einsum("ij,ijk->k", bonds, delta_en))
        assert len(en_result) == len(labels)
        return en_result

    def en_difference_chemi(self, atoms=None):
        """Returns a list of electronegativity metrics, squared and summed over
        adsorbate-site bonds.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        labels = ['dist_' + s + '_chemi' for s in electronegativities]
        if atoms is None:
            return labels
        cm = atoms.connectivity
        chemi = atoms.subsets['chemisorbed_atoms']
        site = atoms.subsets['site_atoms']
        bonds = cm[chemi, :][:, site]
        chemi_numbers = atoms.numbers[chemi]
        site_numbers = atoms.numbers[site]
        en_chemi = list_mendeleev_params(chemi_numbers, electronegativities)
        en_site = list_mendeleev_params(site_numbers, electronegativities)
        delta_en = (en_chemi[:, np.newaxis, :] -
                    en_site[np.newaxis, :, :]) ** 2
        en_result = list(np.einsum("ij,ijk->k", bonds, delta_en))
        assert len(en_result) == len(labels)
        return en_result

    def en_difference_active(self, atoms=None):
        """Returns a list of electronegativity metrics, squared and summed over
        adsorbate bonds including those with the surface.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        labels = ['dist_' + s + '_active' for s in electronegativities]
        if atoms is None:
            return labels
        cm = atoms.connectivity
        ads = atoms.subsets['ads_atoms']
        site = atoms.subsets['site_atoms']
        active = ads + site
        bonds = cm[ads, :][:, active]
        active_numbers = atoms.numbers[active]
        ads_numbers = atoms.numbers[ads]
        en_active = list_mendeleev_params(active_numbers, electronegativities)
        en_ads = list_mendeleev_params(ads_numbers, electronegativities)
        delta_en = (en_ads[:, np.newaxis, :] -
                    en_active[np.newaxis, :, :]) ** 2
        en_result = list(np.einsum("ij,ijk->k", bonds, delta_en))
        assert len(en_result) == len(labels)
        return en_result

    def delta_energy(self, atoms=None):
        """Return the contents of
        atoms.info['key_value_pairs']['delta_energy'] as a feature.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        if atoms is None:
            return ['delta_energy']
        else:
            try:
                delta = float(atoms.info['key_value_pairs']['delta_energy'])
            except KeyError:
                delta = np.nan
            return [delta]

    def db_size(self, atoms=None):
        """Return a fingerprint containing the number of layers in the slab,
        the number of surface atoms in the unit cell and
        the adsorbate coverage.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        labels = ['layers', 'size', 'coverage']
        if atoms is None:
            return labels
        else:
            try:
                layers = float(atoms.info['key_value_pairs']['layers'])
            except KeyError:
                layers = np.nan
            try:
                n = float(atoms.info['key_value_pairs']['n'])
            except KeyError:
                n = np.nan
            size = len(atoms.subsets['termination_atoms'])
            coverage = np.true_divide(n, size)
            return layers, size, coverage

    def dbid(self, atoms=None):
        """Return the contents of atoms.info['id'] as a feature.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        if atoms is None:
            return ['id']
        elif 'id' not in atoms.info:
            return [np.nan]
        else:
            return [int(atoms.info['id'])]

    def ctime(self, atoms=None):
        """Return the contents of atoms.info['ctime'] as a feature.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        if atoms is None:
            return ['time_float']
        elif 'ctime' not in atoms.info:
            return [np.nan]
        else:
            return [atoms.info['ctime']]
