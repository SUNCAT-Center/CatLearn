"""Slab adsorbate fingerprint functions for machine learning."""
import numpy as np
from ase.atoms import string2symbols
from ase.data import ground_state_magnetic_moments as gs_magmom
from ase.data import atomic_numbers
from .periodic_table_data import (get_mendeleev_params, n_outer,
                                  list_mendeleev_params,
                                  default_params, get_radius,
                                  electronegativities)
from .neighbor_matrix import connection_matrix
import collections
from .base import BaseGenerator


block2number = {'s': 1,
                'p': 2,
                'd': 3,
                'f': 4}

# Text based feature.
facetdict = {'001': [1.], '0001step': [2.], '100': [3.],
             '110': [4.], '111': [5.], '211': [6.], '311': [7.],
             '532': [8.]}

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


def check_length(labels, result, atoms):
    if len(result) != len(labels):
        msg = str(len(labels)) + '/' + str(len(result)) + \
            ' labels/fingerprint mismatch.'
        if 'id' in atoms.info:
            msg += ' id: ' + str(atoms.info['id'])
        raise AssertionError(msg)


class ConvolutedFingerprintGenerator(BaseGenerator):
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

        super(ConvolutedFingerprintGenerator, self).__init__(**kwargs)

    def conv_bulk(self, atoms=None):
        """Return a fingerprint vector with propeties convoluted over the
        bulk atoms.

        Parameters
        ----------
            atoms : object
        """
        labels = ['atomic_number_bulk_conv_0',
                  'atomic_volume_bulk_conv_0',
                  'boiling_point_bulk_conv_0',
                  'density_bulk_conv_0',
                  'dipole_polarizability_bulk_conv_0',
                  'electron_affinity_bulk_conv_0',
                  'group_id_bulk_conv_0',
                  'lattice_constant_bulk_conv_0',
                  'melting_point_bulk_conv_0',
                  'period_bulk_conv_0',
                  'vdw_radius_bulk_conv_0',
                  'covalent_radius_cordero_bulk_conv_0',
                  'en_allen_bulk_conv_0',
                  'atomic_weight_bulk_conv_0',
                  'atomic_radius_bulk_conv_0',
                  'heat_of_formation_bulk_conv_0',
                  'dft_bulk_modulus_bulk_conv_0',
                  'dft_rhodensity_bulk_conv_0',
                  'dbcenter_bulk_conv_0',
                  'dbfilling_bulk_conv_0',
                  'dbwidth_bulk_conv_0',
                  'dbskew_bulk_conv_0',
                  'dbkurtosis_bulk_conv_0',
                  'oxi_min_bulk_conv_0',
                  'oxi_med_bulk_conv_0',
                  'oxi_max_bulk_conv_0',
                  'block_bulk_conv_0',
                  'ne_outer_bulk_conv_0',
                  'ne_s_bulk_conv_0',
                  'ne_p_bulk_conv_0',
                  'ne_d_bulk_conv_0',
                  'ne_f_bulk_conv_0',
                  'ionenergy_bulk_conv_0',
                  'ground_state_magmom_bulk_conv_0',
                  'atomic_number_bulk_conv_1',
                  'atomic_volume_bulk_conv_1',
                  'boiling_point_bulk_conv_1',
                  'density_bulk_conv_1',
                  'dipole_polarizability_bulk_conv_1',
                  'electron_affinity_bulk_conv_1',
                  'group_id_bulk_conv_1',
                  'lattice_constant_bulk_conv_1',
                  'melting_point_bulk_conv_1',
                  'period_bulk_conv_1',
                  'vdw_radius_bulk_conv_1',
                  'covalent_radius_cordero_bulk_conv_1',
                  'en_allen_bulk_conv_1',
                  'atomic_weight_bulk_conv_1',
                  'atomic_radius_bulk_conv_1',
                  'heat_of_formation_bulk_conv_1',
                  'dft_bulk_modulus_bulk_conv_1',
                  'dft_rhodensity_bulk_conv_1',
                  'dbcenter_bulk_conv_1',
                  'dbfilling_bulk_conv_1',
                  'dbwidth_bulk_conv_1',
                  'dbskew_bulk_conv_1',
                  'dbkurtosis_bulk_conv_1',
                  'oxi_min_bulk_conv_1',
                  'oxi_med_bulk_conv_1',
                  'oxi_max_bulk_conv_1',
                  'block_bulk_conv_1',
                  'ne_outer_bulk_conv_1',
                  'ne_s_bulk_conv_1',
                  'ne_p_bulk_conv_1',
                  'ne_d_bulk_conv_1',
                  'ne_f_bulk_conv_1',
                  'ionenergy_bulk_conv_1',
                  'ground_state_magmom_bulk_conv_1']
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
                connectivity = atoms.connectivity[bulk]
            else:
                raise NotImplementedError("bulk fingerprint.")
            slab_numbers = atoms.numbers
            dat_b = list_mendeleev_params(numbers, params=self.slab_params)
            dat = list_mendeleev_params(slab_numbers, params=self.slab_params)
            result = list(np.sqrt(np.sum(dat_b * dat_b, axis=0))/len(bulk))
            result += [np.sqrt(np.sum([gs_magmom[z]**2 
                             for z in numbers]))/len(bulk)]
            for i in range(dat.shape[1]):
                result += [np.sqrt(np.sum(dat[:, i][:len(connectivity)].reshape(len(connectivity), 1)
                    * (dat[:, i] * connectivity))) / len(bulk)]
            gsm = np.array([gs_magmom[z] for z in atoms.numbers])
            result += [np.sqrt(np.sum(gsm[:len(connectivity)].reshape(len(connectivity), 1)
                        * (gsm * connectivity))) / len(bulk)]
            check_length(labels, result, atoms)
            return result

    def conv_term(self, atoms=None):
        """Return a fingerprint vector with propeties convoluted over the
        bulk atoms.

        Parameters
        ----------
            atoms : object
        """
        labels = ['atomic_number_term_conv_0',
                  'atomic_volume_term_conv_0',
                  'boiling_point_term_conv_0',
                  'density_term_conv_0',
                  'dipole_polarizability_term_conv_0',
                  'electron_affinity_term_conv_0',
                  'group_id_term_conv_0',
                  'lattice_constant_term_conv_0',
                  'melting_point_term_conv_0',
                  'period_term_conv_0',
                  'vdw_radius_term_conv_0',
                  'covalent_radius_cordero_term_conv_0',
                  'en_allen_term_conv_0',
                  'atomic_weight_term_conv_0',
                  'atomic_radius_term_conv_0',
                  'heat_of_formation_term_conv_0',
                  'dft_bulk_modulus_term_conv_0',
                  'dft_rhodensity_term_conv_0',
                  'dbcenter_term_conv_0',
                  'dbfilling_term_conv_0',
                  'dbwidth_term_conv_0',
                  'dbskew_term_conv_0',
                  'dbkurtosis_term_conv_0',
                  'oxi_min_term_conv_0',
                  'oxi_med_term_conv_0',
                  'oxi_max_term_conv_0',
                  'block_term_conv_0',
                  'ne_outer_term_conv_0',
                  'ne_s_term_conv_0',
                  'ne_p_term_conv_0',
                  'ne_d_term_conv_0',
                  'ne_f_term_conv_0',
                  'ionenergy_term_conv_0',
                  'ground_state_magmom_term_conv_0',
                  'atomic_number_term_conv_1',
                  'atomic_volume_term_conv_1',
                  'boiling_point_term_conv_1',
                  'density_term_conv_1',
                  'dipole_polarizability_term_conv_1',
                  'electron_affinity_term_conv_1',
                  'group_id_term_conv_1',
                  'lattice_constant_term_conv_1',
                  'melting_point_term_conv_1',
                  'period_term_conv_1',
                  'vdw_radius_term_conv_1',
                  'covalent_radius_cordero_term_conv_1',
                  'en_allen_term_conv_1',
                  'atomic_weight_term_conv_1',
                  'atomic_radius_term_conv_1',
                  'heat_of_formation_term_conv_1',
                  'dft_term_modulus_term_conv_1',
                  'dft_rhodensity_term_conv_1',
                  'dbcenter_term_conv_1',
                  'dbfilling_term_conv_1',
                  'dbwidth_term_conv_1',
                  'dbskew_term_conv_1',
                  'dbkurtosis_term_conv_1',
                  'oxi_min_term_conv_1',
                  'oxi_med_term_conv_1',
                  'oxi_max_term_conv_1',
                  'block_term_conv_1',
                  'ne_outer_term_conv_1',
                  'ne_s_term_conv_1',
                  'ne_p_term_conv_1',
                  'ne_d_term_conv_1',
                  'ne_f_term_conv_1',
                  'ionenergy_term_conv_1',
                  'ground_state_magmom_term_conv_1']
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
                connectivity = atoms.connectivity[term]
            else:
                raise NotImplementedError("term fingerprint.")
            slab_numbers = atoms.numbers
            dat_t = list_mendeleev_params(numbers, params=self.slab_params)
            dat = list_mendeleev_params(slab_numbers, params=self.slab_params)
            result = list(np.sqrt(np.sum(dat_t * dat_t, axis=0))/len(term))
            result += [np.sqrt(np.sum([gs_magmom[z]**2 
                             for z in numbers]))/len(term)]
            for i in range(dat.shape[1]):
                result += [np.sqrt(np.sum(dat[:, i][:len(connectivity)].reshape(len(connectivity), 1)
                    * (dat[:, i] * connectivity))) / len(term)]
            gsm = np.array([gs_magmom[z] for z in atoms.numbers])
            result += [np.sqrt(np.sum(gsm[:len(connectivity)].reshape(len(connectivity), 1)
                        * (gsm * connectivity))) / len(term)]
            check_length(labels, result, atoms)
            return result

