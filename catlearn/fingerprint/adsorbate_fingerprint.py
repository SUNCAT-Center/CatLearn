"""Slab adsorbate fingerprint functions for machine learning."""
import numpy as np
from ase.atoms import string2symbols
from ase.data import ground_state_magnetic_moments as gs_magmom
from ase.data import atomic_numbers
from .periodic_table_data import (get_mendeleev_params, n_outer,
                                  list_mendeleev_params,
                                  default_params, get_radius,
                                  electronegativities,
                                  block2number)
from .neighbor_matrix import connection_matrix
import collections
from .base import BaseGenerator


default_adsorbate_fingerprinters = ['mean_chemisorbed_atoms',
                                    'count_chemisorbed_fragment',
                                    'count_ads_atoms',
                                    'count_ads_bonds',
                                    'mean_site',
                                    'sum_site',
                                    'mean_surf_ligands',
                                    'term',
                                    'bulk',
                                    'strain',
                                    'en_difference_ads',
                                    'en_difference_chemi']

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


def check_length(labels, result, atoms):
    """Check that two lists have the same length. If not, print an informative
    error message containing a databse id if present.

    Parameters
    ----------
    labels : list
        A list of feature names.
    result : list
        A fingerprint.
    atoms : object
        A single atoms object.
    """
    if len(result) != len(labels):
        msg = str(len(labels)) + '/' + str(len(result)) + \
            ' labels/fingerprint mismatch.'
        if 'id' in atoms.info:
            msg += ' database id: ' + str(atoms.info['id'])
        raise AssertionError(msg)


class AdsorbateFingerprintGenerator(BaseGenerator):
    def __init__(self, **kwargs):
        """Class containing functions for fingerprint generation.

        Parameters
        ----------
        params : list
            An optional list of parameters upon which to generate features.
        """
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
        labels = ['atomic_number_term',
                  'atomic_volume_term',
                  'boiling_point_term',
                  'density_term',
                  'dipole_polarizability_term',
                  'electron_affinity_term',
                  'group_id_term',
                  'lattice_constant_term',
                  'melting_point_term',
                  'period_term',
                  'vdw_radius_term',
                  'covalent_radius_cordero_term',
                  'en_allen_term',
                  'atomic_weight_term',
                  'atomic_radius_term',
                  'heat_of_formation_term',
                  'dft_bulk_modulus_term',
                  'dft_rhodensity_term',
                  'dbcenter_term',
                  'dbfilling_term',
                  'dbwidth_term',
                  'dbskew_term',
                  'dbkurtosis_term',
                  'oxi_min_term',
                  'oxi_med_term',
                  'oxi_max_term',
                  'block_term',
                  'ne_outer_term',
                  'ne_s_term',
                  'ne_p_term',
                  'ne_d_term',
                  'ne_f_term',
                  'ionenergy_term',
                  'ground_state_magmom_term']
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
            check_length(labels, result, atoms)
            return result

    def bulk(self, atoms=None):
        """Return a fingerprint vector with propeties averaged over
        the bulk atoms.

        Parameters
        ----------
            atoms : object
        """
        labels = ['atomic_number_bulk',
                  'atomic_volume_bulk',
                  'boiling_point_bulk',
                  'density_bulk',
                  'dipole_polarizability_bulk',
                  'electron_affinity_bulk',
                  'group_id_bulk',
                  'lattice_constant_bulk',
                  'melting_point_bulk',
                  'period_bulk',
                  'vdw_radius_bulk',
                  'covalent_radius_cordero_bulk',
                  'en_allen_bulk',
                  'atomic_weight_bulk',
                  'atomic_radius_bulk',
                  'heat_of_formation_bulk',
                  'dft_bulk_modulus_bulk',
                  'dft_rhodensity_bulk',
                  'dbcenter_bulk',
                  'dbfilling_bulk',
                  'dbwidth_bulk',
                  'dbskew_bulk',
                  'dbkurtosis_bulk',
                  'block_bulk',
                  'oxi_min_bulk',
                  'oxi_med_bulk',
                  'oxi_max_bulk',
                  'ne_outer_bulk',
                  'ne_s_bulk',
                  'ne_p_bulk',
                  'ne_d_bulk',
                  'ne_f_bulk',
                  'ionenergy_bulk',
                  'ground_state_magmom_bulk']
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
            check_length(labels, result, atoms)
            return result

    def mean_chemisorbed_atoms(self, atoms=None):
        """Function that takes an atoms objects and returns a fingerprint
        vector containing properties of the closest add atom to a surface
        metal atom.

        Parameters
        ----------
            atoms : object
        """
        labels = ['atomic_number_ads1',
                  'atomic_volume_ads1',
                  'boiling_point_ads1',
                  'density_ads1',
                  'dipole_polarizability_ads1',
                  'electron_affinity_ads1',
                  'group_id_ads1',
                  'lattice_constant_ads1',
                  'melting_point_ads1',
                  'period_ads1',
                  'vdw_radius_ads1',
                  'covalent_radius_cordero_ads1',
                  'en_allen_ads1',
                  'atomic_weight_ads1',
                  'atomic_radius_ads1',
                  'heat_of_formation_ads1',
                  'oxi_min_ads1',
                  'oxi_med_ads1',
                  'oxi_max_ads1',
                  'block_ads1',
                  'ne_outer_ads1',
                  'ne_s_ads1',
                  'ne_p_ads1',
                  'ne_d_ads1',
                  'ne_f_ads1',
                  'ionenergy_ads1',
                  'ground_state_magmom_ads1']
        if atoms is None:
            return labels
        else:
            # Get atomic number of alpha adsorbate atom.
            chemisorbed_atoms = atoms.subsets['chemisorbed_atoms']
            numbers = atoms.numbers[chemisorbed_atoms]
            # Import CatLearn data on that element.
            extra_ads_params = ['atomic_radius', 'heat_of_formation',
                                'oxistates', 'block', 'econf', 'ionenergies']
            dat = list_mendeleev_params(numbers, params=default_params +
                                        extra_ads_params)
            result = list(np.nanmean(dat, axis=0))
            result += [np.nanmean([gs_magmom[z] for z in numbers])]
            check_length(labels, result, atoms)
            return result

    def mean_site(self, atoms=None):
        """Function that takes an atoms objects and returns a fingerprint
        vector with properties averaged over the surface metal atoms
        closest to an add atom.

        Parameters
        ----------
            atoms : object
        """
        labels = ['atomic_number_site_av',
                  'atomic_volume_site_av',
                  'boiling_point_site_av',
                  'density_site_av',
                  'dipole_polarizability_site_av',
                  'electron_affinity_site_av',
                  'group_id_site_av',
                  'lattice_constant_site_av',
                  'melting_point_site_av',
                  'period_site_av',
                  'vdw_radius_site_av',
                  'covalent_radius_cordero_site_av',
                  'en_allen_site_av',
                  'atomic_weight_site_av',
                  'atomic_radius_site_av',
                  'heat_of_formation_site_av',
                  'dft_bulk_modulus_site_av',
                  'dft_rhodensity_site_av',
                  'dbcenter_site_av',
                  'dbfilling_site_av',
                  'dbwidth_site_av',
                  'dbskew_site_av',
                  'dbkurtosis_site_av',
                  'oxi_min_site_av',
                  'oxi_med_site_av',
                  'oxi_max_site_av',
                  'block_site_av',
                  'ne_outer_site_av',
                  'ne_s_site_av',
                  'ne_p_site_av',
                  'ne_d_site_av',
                  'ne_f_site_av',
                  'ionenergy_site_av',
                  'ground_state_magmom_site_av']
        if atoms is None:
            return labels
        else:
            numbers = [atoms[j].number for j in atoms.subsets['site_atoms']]
            dat = list_mendeleev_params(numbers, params=self.slab_params)
            result = list(np.nanmean(dat, axis=0))
            result += [np.nanmean([gs_magmom[z] for z in numbers])]
            check_length(labels, result, atoms)
            return result

    def sum_site(self, atoms=None):
        """Function that takes an atoms objects and returns a fingerprint
        vector with properties summed over the surface metal atoms
        closest to an add atom.

        Parameters
        ----------
            atoms : object
        """
        labels = ['atomic_number_site_sum',
                  'atomic_volume_site_sum',
                  'boiling_point_site_sum',
                  'density_site_sum',
                  'dipole_polarizability_site_sum',
                  'electron_affinity_site_sum',
                  'group_id_site_sum',
                  'lattice_constant_site_sum',
                  'melting_point_site_sum',
                  'period_site_sum',
                  'vdw_radius_site_sum',
                  'covalent_radius_cordero_site_sum',
                  'en_allen_site_sum',
                  'atomic_weight_site_sum',
                  'atomic_radius_site_sum',
                  'heat_of_formation_site_sum',
                  'dft_bulk_modulus_site_sum',
                  'dft_rhodensity_site_sum',
                  'dbcenter_site_sum',
                  'dbfilling_site_sum',
                  'dbwidth_site_sum',
                  'dbskew_site_sum',
                  'dbkurtosis_site_sum',
                  'oxi_min_site_sum',
                  'oxi_med_site_sum',
                  'oxi_max_site_sum',
                  'block_site_sum',
                  'ne_outer_site_sum',
                  'ne_s_site_sum',
                  'ne_p_site_sum',
                  'ne_d_site_sum',
                  'ne_f_site_sum',
                  'ionenergy_site_sum',
                  'ground_state_magmom_site_sum']
        if atoms is None:
            return labels
        else:
            numbers = [atoms[j].number for j in atoms.subsets['site_atoms']]
            dat = list_mendeleev_params(numbers, params=self.slab_params)
            result = list(np.nansum(dat, axis=0))
            result += [np.nansum([gs_magmom[z] for z in numbers])]
            check_length(labels, result, atoms)
            return result

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
        labels = ['nn_surf_ligands', 'identnn_surf_ligands',
                  'atomic_number_surf_ligands',
                  'atomic_volume_surf_ligands',
                  'boiling_point_surf_ligands',
                  'density_surf_ligands',
                  'dipole_polarizability_surf_ligands',
                  'electron_affinity_surf_ligands',
                  'group_id_surf_ligands',
                  'lattice_constant_surf_ligands',
                  'melting_point_surf_ligands',
                  'period_surf_ligands',
                  'vdw_radius_surf_ligands',
                  'covalent_radius_cordero_surf_ligands',
                  'en_allen_surf_ligands',
                  'atomic_weight_surf_ligands',
                  'atomic_radius_surf_ligands',
                  'heat_of_formation_surf_ligands',
                  'dft_bulk_modulus_surf_ligands',
                  'dft_density_surf_ligands',
                  'dbcenter_surf_ligands',
                  'dbfilling_surf_ligands',
                  'dbwidth_surf_ligands',
                  'dbskew_surf_ligands',
                  'dbkurtosis_surf_ligands',
                  'oxi_min_surf_ligands',
                  'oxi_med_surf_ligands',
                  'oxi_max_surf_ligands',
                  'block_surf_ligands',
                  'ne_outer_surf_ligands',
                  'ne_s_surf_ligands',
                  'ne_p_surf_ligands',
                  'ne_d_surf_ligands',
                  'ne_f_surf_ligands',
                  'ionenergy_surf_ligands',
                  'ground_state_magmom_surf_ligands']
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
            check_length(labels, result, atoms)
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
        if atoms is None:
            return ['atomic_number_ads_sum',
                    'atomic_volume_ads_sum',
                    'boiling_point_ads_sum',
                    'density_ads_sum',
                    'dipole_polarizability_ads_sum',
                    'electron_affinity_ads_sum',
                    'group_id_ads_sum',
                    'lattice_constant_ads_sum',
                    'melting_point_ads_sum',
                    'period_ads_sum',
                    'vdw_radius_ads_sum',
                    'covalent_radius_cordero_ads_sum',
                    'en_allen_ads_sum',
                    'atomic_weight_ads_sum',
                    'ne_outer_ads_sum',
                    'ne_s_ads_sum',
                    'ne_p_ads_sum',
                    'ne_d_ads_sum',
                    'ne_f_ads_sum',
                    'ionenergy_ads_sum']
        else:
            ads_atoms = atoms.subsets['ads_atoms']
            dat = []
            for a in ads_atoms:
                Z = atoms.numbers[a]
                ads_params = default_params + ['econf', 'ionenergies']
                mnlv = get_mendeleev_params(Z, params=ads_params)
                dat.append(mnlv[:-2] + list(n_outer(mnlv[-2])) +
                           [mnlv[-1]['1']])
            return list(np.nansum(dat, axis=0))

    def ads_av(self, atoms=None):
        """Function that takes an atoms objects and returns a fingerprint
        vector with averages of the atomic properties of the adsorbate.

        Parameters
        ----------
            atoms : object
        """
        if atoms is None:
            return ['atomic_number_ads_av',
                    'atomic_volume_ads_av',
                    'boiling_point_ads_av',
                    'density_ads_av',
                    'dipole_polarizability_ads_av',
                    'electron_affinity_ads_av',
                    'group_id_ads_av',
                    'lattice_constant_ads_av',
                    'melting_point_ads_av',
                    'period_ads_av',
                    'vdw_radius_ads_av',
                    'covalent_radius_cordero_ads_av',
                    'en_allen_ads_av',
                    'atomic_weight_ads_av',
                    'ne_outer_ads_av',
                    'ne_s_ads_av',
                    'ne_p_ads_av',
                    'ne_d_ads_av',
                    'ne_f_ads_av',
                    'ionenergy_ads_av']
        else:
            ads_atoms = atoms.subsets['ads_atoms']
            dat = []
            for a in ads_atoms:
                Z = int(atoms.numbers[a])
                ads_params = default_params + ['econf', 'ionenergies']
                mnlv = get_mendeleev_params(Z, params=ads_params)
                dat.append(mnlv[:-2] + list(n_outer(mnlv[-2])) +
                           [mnlv[-1]['1']])
            return list(np.nanmean(dat, axis=0))

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
            z_chemi = atoms.numbers[atoms.subsets['chemisorbed_atoms']]
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
            rbulk = []
            rterm = []
            rsite = []
            for b in bulk_numbers:
                rbulk.append(get_radius(b))
            for t in term_numbers:
                rterm.append(get_radius(t))
            for z in z_chemi:
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
                    'block_m1',
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
                    'block_m2',
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
                    'block_sum',
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
            f1 = f1[:-3] + [float(block2number[f1[-3]])] + \
                list(n_outer(f1[-2])) + [f1[-1]['1']] + \
                [float(gs_magmom[z1])]
            if z1 == z2:
                f2 = f1
            else:
                f2 = get_mendeleev_params(z2, params=text_params)
                f2 = f2[:-3] + [float(block2number[f2[-3]])] + \
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
            return ['Ef']
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
            layers = float(atoms.info['key_value_pairs']['layers'])
            size = len(atoms.subsets['termination_atoms'])
            coverage = float(atoms.info['key_value_pairs']['n']) / size
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

    def randomfpv(self, atoms=None):
        """ Returns alist of n random numbers. """
        n = 20
        if atoms is None:
            return ['random'] * n
        else:
            return list(np.random.randint(0, 10, size=n))
