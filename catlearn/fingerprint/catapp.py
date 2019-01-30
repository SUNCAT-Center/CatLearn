"""Slab adsorbate fingerprint functions for machine learning."""
import numpy as np
import collections

from ase.symbols import string2symbols
from ase.data import ground_state_magnetic_moments as gs_magmom
from ase.data import atomic_numbers

from catlearn.featurize.periodic_table_data import (get_mendeleev_params,
                                                    n_outer,
                                                    default_params,
                                                    block2number)
from catlearn.featurize.base import BaseGenerator


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
                                    'bag_connections_ads',
                                    'bag_connections_chemi']

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


class CatappFingerprintGenerator(BaseGenerator):
    def __init__(self, **kwargs):
        """Class containing functions for features equivalent to legacy
        CatApp data.

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
        super(CatappFingerprintGenerator, self).__init__(**kwargs)

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
                    # 'dft_bulk_modulus_m1',
                    # 'dft_rhodensity_m1',
                    # 'dbcenter_m1',
                    # 'dbfilling_m1',
                    # 'dbwidth_m1',
                    # 'dbskew_m1',
                    # 'dbkurtosis_m1',
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
                    # 'dft_bulk_modulus_m2',
                    # 'dft_rhodensity_m2',
                    # 'dbcenter_m2',
                    # 'dbfilling_m2',
                    # 'dbwidth_m2',
                    # 'dbskew_m2',
                    # 'dbkurtosis_m1',
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
                    # 'dft_bulk_modulus_sum',
                    # 'dft_rhodensity_sum',
                    # 'dbcenter_sum',
                    # 'dbfilling_sum',
                    # 'dbwidth_sum',
                    # 'dbskew_sum',
                    # 'dbkurtosis_sum',
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
                                            # 'dft_bulk_modulus',
                                            # 'dft_density',
                                            # 'dbcenter',
                                            # 'dbfilling',
                                            # 'dbwidth',
                                            # 'dbskew',
                                            # 'dbkurt',
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

    def name(self, atoms=None):
        """Return a name for a datapoint based on the contents of
        atoms.info['key_value_pairs'].

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
            return ['catapp_name']
        else:
            kvp = atoms.info['key_value_pairs']
            name = kvp['species'] + '*' + kvp['name'] + kvp['facet']
            return [name]
