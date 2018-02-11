# -*- coding: utf-8 -*-
"""
Slab adsorbate fingerprint functions for machine learning

Created on Tue Dec  6 14:09:29 2016

@author: mhangaard

"""
from __future__ import print_function
import numpy as np
from scipy.spatial import distance
from ase.data import ground_state_magnetic_moments as gs_magmom
from .periodic_table_data import (list_mendeleev_params,
                                  get_radius,
                                  default_params)
# from atoml.fingerprint import neighbor_matrix as nm

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


class BulkFingerprintGenerator(object):
    def __init__(self, extra_params=None, skin=0.3):
        """ Class containing functions for fingerprint generation.
        """
        if extra_params is None:
            self.extra_params = default_extra_params
        else:
            self.extra_params = extra_params
        self.skin = skin

    def summation(self, atoms=None):
        """ Returns a fingerprint vector with propeties of the element name
        saved in the atoms.info['key_value_pairs']['bulk'] """
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
                    'block_sum',
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
            result = list(np.nansum(np.array(dat, dtype=float), axis=0))
            result += [np.nansum([gs_magmom[z] for z in numbers])]
            return result

    def average(self, atoms=None):
        """ Returns a fingerprint vector with propeties of the element name
        saved in the atoms.info['key_value_pairs']['bulk'] """
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
                    'block_av',
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
            result = list(np.nanmean(np.array(dat, dtype=float), axis=0))
            result += [np.nanmean([gs_magmom[z] for z in numbers])]
            return result

    def std(self, atoms=None):
        """ Returns a fingerprint vector with propeties of the element name
        saved in the atoms.info['key_value_pairs']['bulk'] """
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
                    'block_std',
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
            result = list(np.nanstd(np.array(dat, dtype=float), axis=0))
            result += [np.nanstd([gs_magmom[z] for z in numbers])]
            return result

    def cation_summation(self, atoms=None):
        """ Returns a fingerprint vector with propeties of the element name
        saved in the atoms.info['key_value_pairs']['bulk'] """
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
                    'block_sum',
                    'ne_outer_sum',
                    'ne_s_sum',
                    'ne_p_sum',
                    'ne_d_sum',
                    'ne_f_sum',
                    'ionenergy_sum',
                    'ground_state_magmom_sum']
        else:
            numbers = atoms.get_atomic_numbers()
            cations = numbers[numbers != 8]
            dat = list_mendeleev_params(cations, params=default_params +
                                        self.extra_params)
            result = list(np.nansum(np.array(dat, dtype=float), axis=0))
            result += [np.nansum([gs_magmom[z] for z in numbers])]
            return result

    def cation_average(self, atoms=None):
        """ Returns a fingerprint vector with propeties of the element name
        saved in the atoms.info['key_value_pairs']['bulk'] """
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
                    'block_av',
                    'ne_outer_av',
                    'ne_s_av',
                    'ne_p_av',
                    'ne_d_av',
                    'ne_f_av',
                    'ionenergy_av',
                    'ground_state_magmom_av']
        else:
            numbers = atoms.get_atomic_numbers()
            cations = numbers[numbers != 8]
            dat = list_mendeleev_params(cations, params=default_params +
                                        self.extra_params)
            result = list(np.nanmean(np.array(dat, dtype=float), axis=0))
            result += [np.nanmean([gs_magmom[z] for z in numbers])]
            return result

    def cation_std(self, atoms=None):
        """ Returns a fingerprint vector with propeties of the element name
        saved in the atoms.info['key_value_pairs']['bulk'] """
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
                    'block_std',
                    'ne_outer_std',
                    'ne_s_std',
                    'ne_p_std',
                    'ne_d_std',
                    'ne_f_std',
                    'ionenergy_std',
                    'ground_state_magmom_std']
        else:
            numbers = atoms.get_atomic_numbers()
            cations = numbers[numbers != 8]
            dat = list_mendeleev_params(cations, params=default_params +
                                        self.extra_params)
            result = list(np.nanstd(np.array(dat, dtype=float), axis=0))
            result += [np.nanstd([gs_magmom[z] for z in numbers])]
            return result

    def igao_counter(self, atoms=None):
        if atoms is None:
            return ['nAl', 'nIn', 'nGa', 'nO', 'n_ions', 'ex_charge']
        else:
            nAl = len([a for a in atoms if a.symbol == 'Al'])
            nIn = len([a for a in atoms if a.symbol == 'In'])
            nGa = len([a for a in atoms if a.symbol == 'Ga'])
            nO = len([a for a in atoms if a.symbol == 'O'])
            ex_charge = -2 * nO + (nAl + nIn + nGa) * 3
            n_ions = nAl + nIn + nGa
            com = np.sum(atoms.get_center_of_mass())
            result = [nAl, nIn, nGa, nO, ex_charge, n_ions, com]
            return result

    def igao_dist(self, atoms=None):
        if atoms is None:
            return ['d_InO_sum', 'd_GaO_sum', 'd_AlO_sum',
                    'd_InO_av', 'd_GaO_av', 'd_AlO_av',
                    'd_InO_std', 'd_GaO_std', 'd_AlO_std',
                    'd_InO_min', 'd_GaO_min', 'd_AlO_min',
                    'd_InO_max', 'd_GaO_max', 'd_AlO_max',
                    'hexahedality_In', 'hexahedality_Ga', 'hexahedality_Al']
        else:
            dm = atoms.get_all_distances(mic=True)
            # Define cutoff radii for neighbors.
            rO = get_radius(8)
            un = [8, 49, 31, 13]
            radii = np.array([get_radius(z) for z in un])
            cutoffs = radii + rO + self.skin
            # Get indices for each element.
            iO = (atoms.numbers == 8)
            iIn = (atoms.numbers == 49)
            iGa = (atoms.numbers == 31)
            iAl = (atoms.numbers == 13)
            # Get lists of nearest neighbor distances.
            dmO = dm[:, iO]
            dInO = dmO[iIn, :]
            dInOnn = dInO[dInO < cutoffs[1]]
            dGaO = dmO[iGa, :]
            dGaOnn = dGaO[dGaO < cutoffs[2]]
            dAlO = dmO[iAl, :]
            dAlOnn = dAlO[dAlO < cutoffs[3]]
            # Calculate ratio of neighbor distances to averages.
            In_hex = 0.
            Ga_hex = 0.
            Al_hex = 0.
            for i, row in enumerate(dInOnn):
                In_hex += (np.sum(row) / np.average(row))
            for i, row in enumerate(dInOnn):
                Ga_hex += (np.sum(row) / np.average(row))
            for i, row in enumerate(dInOnn):
                Al_hex += (np.sum(row) / np.average(row))
            return [np.sum(dInOnn), np.sum(dGaOnn), np.sum(dAlOnn),
                    np.mean(dInOnn), np.mean(dGaOnn), np.mean(dAlOnn),
                    np.std(dInOnn), np.std(dGaOnn), np.std(dAlOnn),
                    np.min(dInOnn), np.min(dGaOnn), np.min(dAlOnn),
                    np.max(dInOnn), np.max(dGaOnn), np.max(dAlOnn),
                    In_hex, Ga_hex, Al_hex]
    def xyz_id(self, atoms=None):
        if atoms is None:
            return ['xyz_id']
        else:
            return [atoms.info['xyz_id']]
