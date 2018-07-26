"""Slab adsorbate fingerprint functions for machine learning."""
import numpy as np
from ase.atoms import string2symbols
from ase.data import ground_state_magnetic_moments as gs_magmom
from ase.data import atomic_numbers
from .periodic_table_data import (get_mendeleev_params, n_outer,
                                  list_mendeleev_params,
                                  default_params, get_radius,
                                  electronegativities,
                                  block2number, make_labels)
from .neighbor_matrix import connection_matrix
import collections
from .base import BaseGenerator, check_labels


default_adsorbate_fingerprinters = ['mean_chemisorbed_atoms',
                                    'count_chemisorbed_fragment',
                                    'count_ads_atoms',
                                    'count_ads_bonds',
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
                                    'generalized_cn']

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
        if not hasattr(self, 'params'):
            self.slab_params = kwargs.get('params')

        if self.slab_params is None:
            self.slab_params = default_params + extra_slab_params

        if not hasattr(self, 'cn_max'):
            self.cn_max = kwargs.get('cn_max')

        if self.cn_max is None:
            self.cn_max = 12

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

        Returns
        ----------
        result : list
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
        """
        if atoms is None:
            return ['cn_site', 'gcn_site', 'cn_ads1', 'gcn_ads1']
        site = atoms.subsets['site_atoms']
        slab = atoms.subsets['slab_atoms']
        cm = atoms.connectivity
        cn_site = 0.
        for atom in site:
            row = cm[atom, :]
            cn = len([k for i, k in enumerate(row) if k > 0 and i in slab])
            cn_site += cn
            gcn_site = 0.
            for j, btom in enumerate(row):
                if btom > 0 and j in slab:
                    cn = len([k for k in cm[btom, :] if k > 0])
                    gcn_site += btom * cn / self.cn_max

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
                    gcn_chemi += btom * cn / self.cn_max

        return [cn_site / len(site), gcn_site / len(site),
                cn_chemi / len(chemi), gcn_chemi / len(chemi)]

    def count_ads_atoms(self, atoms=None):
        """Function that takes an atoms objects and returns a fingerprint
        vector containing the count of C, O, H and N atoms in the
        adsorbate.

        Parameters
        ----------
            atoms : object
        """
        if atoms is None:
            return ['total_num_H',
                    'total_num_C',
                    'total_num_O',
                    'total_num_N',
                    'total_num_S']
        else:
            nH = len([a.index for a in atoms if a.symbol == 'H'])
            nC = len([a.index for a in atoms if a.symbol == 'C'])
            nO = len([a.index for a in atoms if a.symbol == 'O'])
            nN = len([a.index for a in atoms if a.symbol == 'N'])
            nS = len([a.index for a in atoms if a.symbol == 'S'])
            return [nH, nC, nO, nN, nS]

    def count_chemisorbed_fragment(self, atoms=None):
        """Function that takes an atoms objects and returns a fingerprint
        vector containing the count of C, O, H, N and also metal atoms,
        that are neighbors to the binding atom.
        """
        if atoms is None:
            return ['nn_num_C', 'nn_num_H', 'nn_num_M']
        else:
            chemi = atoms.subsets['chemisorbed_atoms']
            cm = atoms.connectivity
            nH = np.sum(cm[:, chemi] * np.vstack(atoms.numbers == 1))
            nC = np.sum(cm[:, chemi] * np.vstack(atoms.numbers == 6))
            nM = np.sum(cm[:, chemi][atoms.subsets['site_atoms'], :])
            return [nC, nH, nM]

    def mean_surf_ligands(self, atoms=None):
        """Function that takes an atoms objects and returns a fingerprint
        vector containing the count of nearest neighbors and properties of
        the nearest neighbors.

        Parameters
        ----------
            atoms : object
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

    def count_ads_bonds(self, atoms=None):
        """Function that takes an atoms object and returns a fingerprint
        vector with the number of C-H bonds and C-C bonds in the adsorbate.
        The adsorbate atoms must be specified in advance in
        atoms.subsets['ads_atoms']

        Parameters
        ----------
            atoms : object
        """
        if atoms is None:
            return ['nC-C', 'ndouble', 'nC-H', 'nO-H']
        else:
            ads_atoms = atoms[atoms.subsets['ads_atoms']]
            A = connection_matrix(ads_atoms, periodic=True, dx=0.2)
            Hindex = [a.index for a in ads_atoms if a.symbol == 'H']
            Cindex = [a.index for a in ads_atoms if a.symbol == 'C']
            Oindex = [a.index for a in ads_atoms if a.symbol == 'O']
            nCC = 0
            nCH = 0
            nC2 = 0
            nOH = 0
            nOdouble = 0
            nCdouble = 0
            nCtriple = 0
            nCquad = 0
            for o in Oindex:
                nOH += np.sum(A[Hindex, o])
                Onn = np.sum(A[:, o])
                if Onn == 1:
                    nOdouble += 1
            for c in Cindex:
                nCC += np.sum(A[Cindex, c])
                nCH += np.sum(A[Hindex, c])
                Cnn = np.sum(A[:, c])
                if Cnn == 3:
                    nCdouble += 1
                elif Cnn == 2:
                    if nCH > 0:
                        nCtriple += 1
                    else:
                        nCdouble += 2
                elif Cnn == 1:
                    nCquad += 1
                nC2 += 4 - (nCC + nCH)
            return [nCC, nC2, nCH, nOH]

    def ads_sum(self, atoms=None):
        """Function that takes an atoms objects and returns a fingerprint
        vector with averages of the atomic properties of the adsorbate.

        Parameters
        ----------
            atoms : object
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

    def en_difference_ads(self, atoms=None):
        """Returns a list of electronegativity metrics, squared and summed over
        bonds within the adsorbate atoms.

        Parameters
        ----------
            atoms : object
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

    def catapp_AB(self, atoms=None):
        if atoms is None:
            return ['atomic_number_m1',
                    'atomic_volume_m1',
                    'boiling_point_m1',
                    'density_m1',
                    'dipole_polarizability_m1',
                    'electron_affinity_m1',
                    'group_id_m1',
                    'lattice_constant_m1',
                    'melting_point_m1',
                    'period_m1',
                    'vdw_radius_m1',
                    'covalent_radius_cordero_m1',
                    'en_allen_m1',
                    'atomic_weight_m1',
                    'heat_of_formation_m1',
                    #'dft_bulk_modulus_m1',
                    #'dft_rhodensity_m1',
                    #'dbcenter_m1',
                    #'dbfilling_m1',
                    #'dbwidth_m1',
                    #'dbskew_m1',
                    #'dbkurtosis_m1',
                    'sblock_m1',
                    'pblock_m1',
                    'dblock_m1',
                    'fblock_m1',
                    'ne_outer_m1',
                    'ne_s_m1',
                    'ne_p_m1',
                    'ne_d_m1',
                    'ne_f_m1',
                    'ionenergy_m1',
                    'ground_state_magmom_m1',
                    'atomic_number_m2',
                    'atomic_volume_m2',
                    'boiling_point_m2',
                    'density_m2',
                    'dipole_polarizability_m2',
                    'electron_affinity_m2',
                    'group_id_m2',
                    'lattice_constant_m2',
                    'melting_point_m2',
                    'period_m2',
                    'vdw_radius_m2',
                    'covalent_radius_cordero_m2',
                    'en_allen_m2',
                    'atomic_weight_m2',
                    'heat_of_formation_m2',
                    #'dft_bulk_modulus_m2',
                    #'dft_rhodensity_m2',
                    #'dbcenter_m2',
                    #'dbfilling_m2',
                    #'dbwidth_m2',
                    #'dbskew_m2',
                    #'dbkurtosis_m1',
                    'sblock_m2',
                    'pblock_m2',
                    'dblock_m2',
                    'fblock_m2',
                    'ne_outer_m2',
                    'ne_s_m2',
                    'ne_p_m2',
                    'ne_d_m2',
                    'ne_f_m2',
                    'ionenergy_m2',
                    'ground_state_magmom_m2',
                    'atomic_number_sum',
                    'atomic_volume_sum',
                    'boiling_point_sum',
                    'density_sum',
                    'dipole_polarizability_sum',
                    'electron_affinity_sum',
                    'group_id_sum',
                    'lattice_constant_sum',
                    'melting_point_sum',
                    'period_sum',
                    'vdw_radius_sum',
                    'covalent_radius_cordero_sum',
                    'en_allen_sum',
                    'atomic_weight_sum',
                    'heat_of_formation_sum',
                    #'dft_bulk_modulus_sum',
                    #'dft_rhodensity_sum',
                    #'dbcenter_sum',
                    #'dbfilling_sum',
                    #'dbwidth_sum',
                    #'dbskew_sum',
                    #'dbkurtosis_sum',
                    'sblock_sum',
                    'pblock_sum',
                    'dblock_sum',
                    'fblock_sum',
                    'ne_outer_sum',
                    'ne_s_sum',
                    'ne_p_sum',
                    'ne_d_sum',
                    'ne_f_sum',
                    'ionenergy_sum',
                    'ground_state_magmom_sum',
                    'concentration_catapp',
                    'facet_catapp',
                    'site_catapp']
        else:
            # Atomic numbers in the site.
            Z_surf1_raw = [atoms.numbers[j]
                           for j in atoms.subsets['ligand_atoms']]
            # Sort by concentration
            counts = collections.Counter(Z_surf1_raw)
            Z_surf1 = sorted(Z_surf1_raw, key=counts.get, reverse=True)
            z1 = Z_surf1[0]
            z2 = Z_surf1[0]
            for z in Z_surf1:
                if z != z1:
                    z2 = z
            uu, ui = np.unique(Z_surf1, return_index=True)
            if len(ui) == 1:
                if Z_surf1[0] == z1:
                    site = 1.
                elif Z_surf1[0] == z2:
                    site = 3.
            else:
                site = 2.
            # Import overlayer composition from ase database.
            kvp = atoms.info['key_value_pairs']
            term = [atomic_numbers[zt] for zt in string2symbols(kvp['term'])]
            termuu, termui = np.unique(term, return_index=True)
            if '3' in kvp['term']:
                conc = 3.
            elif len(termui) == 1:
                conc = 1.
            elif len(termui) == 2:
                conc = 2.
            else:
                raise NotImplementedError("catappAB only supports AxBy.")
            text_params = default_params + ['heat_of_formation',
                                            #'dft_bulk_modulus',
                                            #'dft_density',
                                            #'dbcenter',
                                            #'dbfilling',
                                            #'dbwidth',
                                            #'dbskew',
                                            #'dbkurt',
                                            'block',
                                            'econf',
                                            'ionenergies']
            f1 = get_mendeleev_params(z1, params=text_params)
            f1 = f1[:-3] + block2number[f1[-3]] + \
                list(n_outer(f1[-2])) + [f1[-1]['1']] + \
                [float(gs_magmom[z1])]
            if z1 == z2:
                f2 = f1
            else:
                f2 = get_mendeleev_params(z2, params=text_params)
                f2 = f2[:-3] + block2number[f2[-3]] + \
                    list(n_outer(f2[-2])) + [f2[-1]['1']] + \
                    [float(gs_magmom[z2])]
            msum = list(np.nansum([f1, f2], axis=0, dtype=np.float))
            facet = facetdict[kvp['facet'].replace(')', '').replace('(', '')]
            fp = f1 + f2 + msum + [conc] + facet + [site]
            return fp

    def delta_energy(self, atoms=None):
        """Return the contents of
        atoms.info['key_value_pairs']['delta_energy'] as a feature.

        Parameters
        ----------
            atoms : object
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

    def name(self, atoms=None):
        """Return a name for a datapoint based on the contents of
        atoms.info['key_value_pairs'].

        Parameters
        ----------
            atoms : object
        """
        if atoms is None:
            return ['catapp_name']
        else:
            kvp = atoms.info['key_value_pairs']
            name = kvp['species'] + '*' + kvp['name'] + kvp['facet']
            return [name]

    def dbid(self, atoms=None):
        """Return the contents of atoms.info['id'] as a feature.

        Parameters
        ----------
            atoms : object
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
        """
        if atoms is None:
            return ['time_float']
        elif 'ctime' not in atoms.info:
            return [np.nan]
        else:
            return [atoms.info['ctime']]
