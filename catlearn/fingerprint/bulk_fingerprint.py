"""Slab adsorbate fingerprint functions for machine learning."""
from __future__ import print_function
import numpy as np

from ase.data import ground_state_magnetic_moments as gs_magmom
from ase.data import atomic_numbers

from .periodic_table_data import (list_mendeleev_params,
                                  get_radius,
                                  default_params)
from .base import BaseGenerator

default_extra_params = ['c6',
                        'c6_gb',
                        'en_allred-rochow',
                        'en_cottrell-sutton',
                        'en_gordy',
                        'en_martynov-batsanov',
                        'en_mulliken',
                        'en_nagle',
                        'en_ghosh',
                        'gas_basicity',
                        'metallic_radius_c12',
                        'atomic_radius',
                        'heat_of_formation',
                        'dft_std_modulus',
                        'dft_density',
                        'en_pauling',
                        'dbcenter',
                        'dbfilling',
                        'dbwidth',
                        'dbskew',
                        'dbkurt',
                        'block',
                        'econf',
                        'ionenergies']


default_bulk_fingerprinters = ['bulk_summation',
                               'bulk_average',
                               'bulk_std']


class BulkFingerprintGenerator(BaseGenerator):
    def __init__(self, **kwargs):
        """Class containing functions for fingerprint generation."""
        if not hasattr(self, 'extra_params'):
            self.extra_params = kwargs.get('extra_params', None)
        if self.extra_params is None:
            self.extra_params = default_extra_params
        if not hasattr(self, 'skin'):
            self.skin = kwargs.get('skin', 0.2)
        # Ceramic datasets can rely on fingerprints of the metal ion only.
        if not hasattr(self, 'ceramic_element'):
            self.ceramic_element = kwargs.get('ceramic_element', None)
        if isinstance(self.ceramic_element, str):
            self.ceramic_element = atomic_numbers[self.ceramic_element]

        super(BulkFingerprintGenerator, self).__init__(**kwargs)

    def bulk_summation(self, atoms=None):
        """Return a fingerprint vector with propeties of the element name
        saved in the atoms.info['key_value_pairs']['bulk']"""
        if atoms is None:
            return ['atomic_number_sum',
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
                    'c6_sum',
                    'c6_gb_sum',
                    'en_allred-rochow_sum',
                    'en_cottrell-sutton_sum',
                    'en_gordy_sum',
                    'en_martynov-batsanov_sum',
                    'en_mulliken_sum',
                    'en_nagle_sum',
                    'en_ghosh_sum',
                    'gas_basicity_sum',
                    'metallic_radius_sum',
                    'atomic_radius_sum',
                    'heat_of_formation_sum',
                    'dft_sum_modulus_sum',
                    'dft_rhodensity_sum',
                    'en_pauling_sum',
                    'dbcenter_sum',
                    'dbfilling_sum',
                    'dbwidth_sum',
                    'dbskew_sum',
                    'dbkurtosis_sum',
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
                    'ground_state_magmom_sum']
        else:
            numbers = atoms.get_atomic_numbers()
            dat = list_mendeleev_params(numbers, params=default_params +
                                        self.extra_params)
            result = list(np.nansum(dat, axis=0))
            result += [np.nansum([gs_magmom[z] for z in numbers])]
            return result

    def bulk_average(self, atoms=None):
        """Return a fingerprint vector with propeties of the element name
        saved in the atoms.info['key_value_pairs']['bulk']"""
        if atoms is None:
            return ['atomic_number_av',
                    'atomic_volume_av',
                    'boiling_point_av',
                    'density_av',
                    'dipole_polarizability_av',
                    'electron_affinity_av',
                    'group_id_av',
                    'lattice_constant_av',
                    'melting_point_av',
                    'period_av',
                    'vdw_radius_av',
                    'covalent_radius_cordero_av',
                    'en_allen_av',
                    'atomic_weight_av',
                    'c6_av',
                    'c6_gb_av',
                    'en_allred-rochow_av',
                    'en_cottrell-sutton_av',
                    'en_gordy',
                    'en_martynov-batsanov_av',
                    'en_mulliken_av',
                    'en_nagle_av',
                    'en_ghosh_av',
                    'gas_basicity_av',
                    'metallic_radius_av',
                    'atomic_radius_av',
                    'heat_of_formation_av',
                    'dft_av_modulus_av',
                    'dft_rhodensity_av',
                    'en_pauling_av',
                    'dbcenter_av',
                    'dbfilling_av',
                    'dbwidth_av',
                    'dbskew_av',
                    'dbkurtosis_av',
                    'sblock_av',
                    'pblock_av',
                    'dblock_av',
                    'fblock_av',
                    'ne_outer_av',
                    'ne_s_av',
                    'ne_p_av',
                    'ne_d_av',
                    'ne_f_av',
                    'ionenergy_av',
                    'ground_state_magmom_av']
        else:
            numbers = atoms.get_atomic_numbers()
            dat = list_mendeleev_params(numbers, params=default_params +
                                        self.extra_params)
            result = list(np.nanmean(dat, axis=0))
            result += [np.nanmean([gs_magmom[z] for z in numbers])]
            return result

    def bulk_std(self, atoms=None):
        """Return a fingerprint vector with propeties of the element name
        saved in the atoms.info['key_value_pairs']['bulk']"""
        if atoms is None:
            return ['atomic_number_std',
                    'atomic_volume_std',
                    'boiling_point_std',
                    'density_std',
                    'dipole_polarizability_std',
                    'electron_affinity_std',
                    'group_id_std',
                    'lattice_constant_std',
                    'melting_point_std',
                    'period_std',
                    'vdw_radius_std',
                    'covalent_radius_cordero_std',
                    'en_allen_std',
                    'atomic_weight_std',
                    'c6_std',
                    'c6_gb_std',
                    'en_allred-rochow_std',
                    'en_cottrell-sutton_std',
                    'en_gordy',
                    'en_martynov-batsanov_std',
                    'en_mulliken_std',
                    'en_nagle_std',
                    'en_ghosh_std',
                    'gas_basicity_std',
                    'metallic_radius_std',
                    'atomic_radius',
                    'heat_of_formation_std',
                    'dft_std_modulus_std',
                    'dft_rhodensity_std',
                    'en_pauling_std',
                    'dbcenter_std',
                    'dbfilling_std',
                    'dbwidth_std',
                    'dbskew_std',
                    'dbkurtosis_std',
                    'sblock_std',
                    'pblock_std',
                    'dblock_std',
                    'fblock_std',
                    'ne_outer_std',
                    'ne_s_std',
                    'ne_p_std',
                    'ne_d_std',
                    'ne_f_std',
                    'ionenergy_std',
                    'ground_state_magmom_std']
        else:
            numbers = atoms.get_atomic_numbers()
            dat = list_mendeleev_params(numbers, params=default_params +
                                        self.extra_params)
            result = list(np.nanstd(dat, axis=0))
            result += [np.nanstd([gs_magmom[z] for z in numbers])]
            return result

    def ceramic_summation(self, atoms=None):
        """Return a fingerprint vector with propeties of the element name
        saved in the atoms.info['key_value_pairs']['bulk']"""
        if atoms is None:
            return ['atomic_number_sum',
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
                    'c6_sum',
                    'c6_gb_sum',
                    'en_allred-rochow_sum',
                    'en_cottrell-sutton_sum',
                    'en_gordy_sum',
                    'en_martynov-batsanov_sum',
                    'en_mulliken_sum',
                    'en_nagle_sum',
                    'en_ghosh_sum',
                    'gas_basicity_sum',
                    'metallic_radius_sum',
                    'atomic_radius_sum',
                    'heat_of_formation_sum',
                    'dft_sum_modulus_sum',
                    'dft_rhodensity_sum',
                    'en_pauling_sum',
                    'dbcenter_sum',
                    'dbfilling_sum',
                    'dbwidth_sum',
                    'dbskew_sum',
                    'dbkurtosis_sum',
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
                    'ground_state_magmom_sum']
        else:
            numbers = atoms.get_atomic_numbers()
            cations = numbers[numbers != self.ceramic_element]
            dat = list_mendeleev_params(cations, params=default_params +
                                        self.extra_params)
            result = list(np.nansum(dat, axis=0))
            result += [np.nansum([gs_magmom[z] for z in numbers])]
            return result

    def ceramic_average(self, atoms=None):
        """Return a fingerprint vector with propeties of the element name
        saved in the atoms.info['key_value_pairs']['bulk']"""
        if atoms is None:
            return ['atomic_number_av',
                    'atomic_volume_av',
                    'boiling_point_av',
                    'density_av',
                    'dipole_polarizability_av',
                    'electron_affinity_av',
                    'group_id_av',
                    'lattice_constant_av',
                    'melting_point_av',
                    'period_av',
                    'vdw_radius_av',
                    'covalent_radius_cordero_av',
                    'en_allen_av',
                    'atomic_weight_av',
                    'c6_av',
                    'c6_gb_av',
                    'en_allred-rochow_av',
                    'en_cottrell-sutton_av',
                    'en_gordy',
                    'en_martynov-batsanov_av',
                    'en_mulliken_av',
                    'en_nagle_av',
                    'en_ghosh_av',
                    'gas_basicity_av',
                    'metallic_radius_av',
                    'atomic_radius_av',
                    'heat_of_formation_av',
                    'dft_av_modulus_av',
                    'dft_rhodensity_av',
                    'en_pauling_av',
                    'dbcenter_av',
                    'dbfilling_av',
                    'dbwidth_av',
                    'dbskew_av',
                    'dbkurtosis_av',
                    'sblock_av',
                    'pblock_av',
                    'dblock_av',
                    'fblock_av',
                    'ne_outer_av',
                    'ne_s_av',
                    'ne_p_av',
                    'ne_d_av',
                    'ne_f_av',
                    'ionenergy_av',
                    'ground_state_magmom_av']
        else:
            numbers = atoms.get_atomic_numbers()
            cations = numbers[numbers != self.ceramic_element]
            dat = list_mendeleev_params(cations, params=default_params +
                                        self.extra_params)
            result = list(np.nanmean(dat, axis=0))
            result += [np.nanmean([gs_magmom[z] for z in numbers])]
            return result

    def ceramic_std(self, atoms=None):
        """Return a fingerprint vector with propeties of the element name
        saved in the atoms.info['key_value_pairs']['bulk']"""
        if atoms is None:
            return ['atomic_number_std',
                    'atomic_volume_std',
                    'boiling_point_std',
                    'density_std',
                    'dipole_polarizability_std',
                    'electron_affinity_std',
                    'group_id_std',
                    'lattice_constant_std',
                    'melting_point_std',
                    'period_std',
                    'vdw_radius_std',
                    'covalent_radius_cordero_std',
                    'en_allen_std',
                    'atomic_weight_std',
                    'c6_std',
                    'c6_gb_std',
                    'en_allred-rochow_std',
                    'en_cottrell-sutton_std',
                    'en_gordy',
                    'en_martynov-batsanov_std',
                    'en_mulliken_std',
                    'en_nagle_std',
                    'en_ghosh_std',
                    'gas_basicity_std',
                    'metallic_radius_std',
                    'atomic_radius',
                    'heat_of_formation_std',
                    'dft_std_modulus_std',
                    'dft_rhodensity_std',
                    'en_pauling_std',
                    'dbcenter_std',
                    'dbfilling_std',
                    'dbwidth_std',
                    'dbskew_std',
                    'dbkurtosis_std',
                    'sblock_std',
                    'pblock_std',
                    'dblock_std',
                    'fblock_std',
                    'ne_outer_std',
                    'ne_s_std',
                    'ne_p_std',
                    'ne_d_std',
                    'ne_f_std',
                    'ionenergy_std',
                    'ground_state_magmom_std']
        else:
            numbers = atoms.get_atomic_numbers()
            cations = numbers[numbers != self.ceramic_element]
            dat = list_mendeleev_params(cations, params=default_params +
                                        self.extra_params)
            result = list(np.nanstd(dat, axis=0))
            result += [np.nanstd([gs_magmom[z] for z in numbers])]
            return result

    def ceramic_counter(self, atoms=None):
        if atoms is None:
            return ['n_ions', 'n_ceramic']
        else:
            n_ceramic = len([a for a in atoms if a.number == self.ceramic])
            n_ions = len(atoms) - n_ceramic
            # Append oxidation state or excess charge fingerprint here.
            return [n_ions, n_ceramic]

    def ceramic_dist(self, atoms=None):
        if atoms is None:
            return ['d_ion-ceramic_sum', 'd_ion-ceramic_av',
                    'd_ion-ceramic_std',
                    'd_ion-ceramic_min', 'd_ion-ceramic_max']
        else:
            dm = atoms.get_all_distances(mic=True)
            # Define cutoff radii for neighbors.
            r_ceramic = get_radius(self.ceramic_element)
            all_radii = np.array([get_radius(z) for z, s in
                                  enumerate(atomic_numbers) if
                                  z > 0 and z < 93])
            all_cutoffs = all_radii + r_ceramic + self.skin
            # Get indices for each element.
            i_ceramic = np.where(atoms.numbers == self.ceramic_element)[0]
            i_ion = np.where(atoms.numbers != self.ceramic_element)[0]
            # Get lists of atomic distances to oxygens.
            dm_ceramic = dm[:, i_ceramic]
            # Get lists of nearest neighbor distances.
            ion_numbers = atoms.numbers[i_ion]
            d_ion_a = dm[i_ion, :]
            i_ion_nn = (d_ion_a < np.vstack(all_cutoffs[ion_numbers]))
            d_ion_nn = dm_ceramic[i_ion_nn]
            result = [np.nansum(d_ion_nn), np.nanmean(d_ion_nn),
                      np.nanstd(d_ion_nn),
                      np.nanmin(d_ion_nn), np.nanmax(d_ion_nn)]
            return result

    def xyz_id(self, atoms=None):
        if atoms is None:
            return ['xyz_id']
        else:
            return [atoms.info['xyz_id']]
