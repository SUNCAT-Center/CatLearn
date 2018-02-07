# -*- coding: utf-8 -*-
"""
Slab adsorbate fingerprint functions for machine learning

Created on Tue Dec  6 14:09:29 2016

@author: mhangaard

"""
from __future__ import print_function
import numpy as np
from ase.data import ground_state_magnetic_moments as gs_magmom
from .periodic_table_data import (average_mendeleev_params,
                                  default_params)

default_extra_params = ['heat_of_formation',
                        'dft_bulk_modulus',
                        'dft_density',
                        'dbcenter',
                        'dbfilling',
                        'dbwidth',
                        'dbskew',
                        'dbkurt',
                        'block',
                        'econf',
                        'ionenergies']


class BulkFingerprintGenerator(object):
    def __init__(self, extra_params=None):
        """ Class containing functions for fingerprint generation.
        """
        if extra_params is None:
            self.extra_params = default_extra_params
        else:
            self.extra_params = extra_params

    def bulk(self, atoms=None):
        """ Returns a fingerprint vector with propeties of the element name
        saved in the atoms.info['key_value_pairs']['bulk'] """
        if atoms is None:
            return ['atomic_number_bulk',
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
                    'heat_of_formation_bulk',
                    'block_bulk',
                    'dft_bulk_modulus_bulk',
                    'dft_rhodensity_bulk',
                    'dbcenter_bulk',
                    'dbfilling_bulk',
                    'dbwidth_bulk',
                    'dbskew_bulk',
                    'dbkurtosis_bulk',
                    'ne_outer_bulk',
                    'ne_s_bulk',
                    'ne_p_bulk',
                    'ne_d_bulk',
                    'ne_f_bulk',
                    'ionenergy_bulk',
                    'ground_state_magmom_bulk']
        else:
            numbers = atoms.get_atomic_numbers()
            result = average_mendeleev_params(numbers,
                                              params=default_params +
                                              self.extra_params)
            result += [np.mean([gs_magmom[z] for z in numbers])]
            return result
