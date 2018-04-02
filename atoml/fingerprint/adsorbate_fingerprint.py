# -*- coding: utf-8 -*-
"""
Slab adsorbate fingerprint functions for machine learning

Created on Tue Dec  6 14:09:29 2016

@author: mhangaard

"""
from __future__ import print_function

import numpy as np
from ase.atoms import string2symbols
from ase.data import ground_state_magnetic_moments as gs_magmom
from ase.data import covalent_radii, atomic_numbers
from .periodic_table_data import (get_mendeleev_params, n_outer,
                                  list_mendeleev_params,
                                  default_params, get_radius)
from .adsorbate_prep import layers_info
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
        """ Returns a fingerprint vector with propeties of the element name
        saved in the atoms.info['key_value_pairs']['term'] """
        if atoms is None:
            return ['atomic_number_term',
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
        else:
            if ('key_value_pairs' in atoms.info and
                    'term' in atoms.info['key_value_pairs']):
                term = atoms.info['key_value_pairs']['term']
            elif 'termination' in atoms.info:
                term = atoms.info['termination_atoms']
            else:
                raise NotImplementedError("termination fingerprint.")
            # A = float(atoms.cell[0, 0]) * float(atoms.cell[1, 1])
            numbers = [atomic_numbers[s] for s in string2symbols(term)]
            dat = list_mendeleev_params(numbers, params=self.slab_params)
            result = list(np.nanmean(np.array(dat, dtype=float), axis=0))
            result += [np.nanmean([gs_magmom[z] for z in numbers])]
            return result

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
        else:
            if ('key_value_pairs' in atoms.info and
                    'bulk' in atoms.info['key_value_pairs']):
                bulk = atoms.info['key_value_pairs']['bulk']
            elif 'bulk_atoms' in atoms.info:
                bulk = atoms.info['bulk_atoms']
            else:
                raise NotImplementedError("bulk fingerprint.")
            numbers = [atomic_numbers[s] for s in string2symbols(bulk)]
            dat = list_mendeleev_params(numbers, params=self.slab_params)
            result = list(np.nanmean(np.array(dat, dtype=float), axis=0))
            result += [np.nanmean([gs_magmom[z] for z in numbers])]
            return result

    def primary_addatom(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector containing properties of the closest add atom to a surface
            metal atom.
        """
        if atoms is None:
            return ['atomic_number_ads1',
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
        else:
            # Get atomic number of alpha adsorbate atom.
            chemisorbed_atoms = atoms.info['chemisorbed_atoms']
            numbers = atoms.numbers[chemisorbed_atoms]
            # Import AtoML data on that element.
            extra_ads_params = ['atomic_radius', 'heat_of_formation',
                                'oxistates', 'block', 'econf', 'ionenergies']
            dat = list_mendeleev_params(numbers, params=default_params +
                                        extra_ads_params)
            result = list(np.nanmean(np.array(dat, dtype=float), axis=0))
            result += [np.nanmean([gs_magmom[z] for z in numbers])]
            return result

    def primary_surfatom(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector with properties averaged over the surface metal atoms
            closest to an add atom.
        """
        if atoms is None:
            return ['atomic_number_surf1av',
                    'atomic_volume_surf1av',
                    'boiling_point_surf1av',
                    'density_surf1av',
                    'dipole_polarizability_surf1av',
                    'electron_affinity_surf1av',
                    'group_id_surf1av',
                    'lattice_constant_surf1av',
                    'melting_point_surf1av',
                    'period_surf1av',
                    'vdw_radius_surf1av',
                    'covalent_radius_cordero_surf1av',
                    'en_allen_surf1av',
                    'atomic_weight_surf1av',
                    'atomic_radius_surf1av',
                    'heat_of_formation_surf1av',
                    'dft_bulk_modulus_surf1av',
                    'dft_rhodensity_surf1av',
                    'dbcenter_surf1av',
                    'dbfilling_surf1av',
                    'dbwidth_surf1av',
                    'dbskew_surf1av',
                    'dbkurtosis_surf1av',
                    'oxi_min_surf1av',
                    'oxi_med_surf1av',
                    'oxi_max_surf1av',
                    'block_surf1av',
                    'ne_outer_surf1av',
                    'ne_s_surf1av',
                    'ne_p_surf1av',
                    'ne_d_surf1av',
                    'ne_f_surf1av',
                    'ionenergy_surf1av',
                    'ground_state_magmom_surf1av']
        else:
            numbers = [atoms[j].number for j in atoms.info['site_atoms']]
            dat = list_mendeleev_params(numbers, params=self.slab_params)
            result = list(np.nanmean(np.array(dat, dtype=float), axis=0))
            result += [np.nanmean([gs_magmom[z] for z in numbers])]
            return result

    def primary_surfatom_sum(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector with properties summed over the surface metal atoms
            closest to an add atom.
        """
        if atoms is None:
            return ['atomic_number_surf1sum',
                    'atomic_volume_surf1sum',
                    'boiling_point_surf1sum',
                    'density_surf1sum',
                    'dipole_polarizability_surf1sum',
                    'electron_affinity_surf1sum',
                    'group_id_surf1sum',
                    'lattice_constant_surf1sum',
                    'melting_point_surf1sum',
                    'period_surf1sum',
                    'vdw_radius_surf1sum',
                    'covalent_radius_cordero_surf1sum',
                    'en_allen_surf1sum',
                    'atomic_weight_surf1sum',
                    'atomic_radius_surf1sum',
                    'heat_of_formation_surf1sum',
                    'dft_bulk_modulus_surf1sum',
                    'dft_rhodensity_surf1sum',
                    'dbcenter_surf1sum',
                    'dbfilling_surf1sum',
                    'dbwidth_surf1sum',
                    'dbskew_surf1sum',
                    'dbkurtosis_surf1sum',
                    'oxi_min_surf1sum',
                    'oxi_med_surf1sum',
                    'oxi_max_surf1sum',
                    'block_surf1sum',
                    'ne_outer_surf1sum',
                    'ne_s_surf1sum',
                    'ne_p_surf1sum',
                    'ne_d_surf1sum',
                    'ne_f_surf1sum',
                    'ionenergy_surf1sum',
                    'ground_state_magmom_surf1sum']
        else:
            numbers = [atoms[j].number for j in atoms.info['site_atoms']]
            dat = list_mendeleev_params(numbers, params=self.slab_params)
            result = list(np.nansum(np.array(dat, dtype=float), axis=0))
            result += [np.nansum([gs_magmom[z] for z in numbers])]
            return result

    def Z_add(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector containing the count of C, O, H and N atoms in the
            adsorbate.
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
            return [nC, nO, nH, nS, nN]

    def primary_adds_nn(self, atoms=None, rtol=1.15):
        """ Function that takes an atoms objects and returns a fingerprint
            vector containing the count of C, O, H, N and also metal atoms,
            that are neighbors to the binding atom.
        """
        if atoms is None:
            return ['nn_num_C', 'nn_num_H', 'nn_num_M']
        else:
            chemi = atoms.info['chemisorbed_atoms']
            nl = atoms.get_neighborlist()
            nH = []
            nC = []
            for i in chemi:
                for b in nl[i]:
                    if atoms.numbers[b] == 1:
                        nH.append(b)
                    elif atoms.numbers[b] == 6:
                        nC.append(b)
            nM = len(atoms.info['ligand_atoms'])
            return [nC, nH, nM]

    def secondary_adds_nn(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector containing the count of C, O, H and N atoms in the adsorbate
            second group.

            This function is relevant for adsorbates that bind through 2 atoms.
            Example: CH2CH2 on Pt111 binds through 2 C atoms
        """
        if atoms is None:
            return ['num_C_add2nn', 'num_H_add2nn', 'num_M_add2nn']
        else:
            slab_atoms = atoms.info['slab_atoms']
            ads_atoms = atoms.info['ads_atoms']
            liste = []
            for m in slab_atoms:
                for a in ads_atoms:
                    d = atoms.get_distance(m, a, mic=True, vector=False)
                    liste.append([a, d])
            L = np.array(liste)
            i = np.argsort(L[:, 1])[1]
            secondary_add = int(L[i, 0])
            nH1 = len([a.index for a in atoms if a.symbol == 'H' and
                       atoms.get_distance(secondary_add, a.index, mic=True) <
                       1.15 and a.index != secondary_add])
            nC1 = len([a.index for a in atoms if a.symbol == 'C' and
                       atoms.get_distance(secondary_add, a.index, mic=True) <
                       1.15 and a.index != secondary_add])
            nM = len([a.index for a in atoms if a.symbol not in ads_atoms and
                      atoms.get_distance(secondary_add, a.index, mic=True) <
                      2.35])
            return [nC1, nH1, nM]  # , nN, nH]

    def primary_surf_nn(self, atoms=None, rtol=1.3):
        """ Function that takes an atoms objects and returns a fingerprint
            vector containing the count of nearest neighbors and properties of
            the nearest neighbors.
        """
        if atoms is None:
            return ['nn_surf2', 'identnn_surf2',
                    'atomic_number_surf2',
                    'atomic_volume_surf2',
                    'boiling_point_surf2',
                    'density_surf2',
                    'dipole_polarizability_surf2',
                    'electron_affinity_surf2',
                    'group_id_surf2',
                    'lattice_constant_surf2',
                    'melting_point_surf2',
                    'period_surf2',
                    'vdw_radius_surf2',
                    'covalent_radius_cordero_surf2',
                    'en_allen_surf2',
                    'atomic_weight_surf2',
                    'atomic_radius_surf2',
                    'heat_of_formation_surf2',
                    'dft_bulk_modulus_surf2',
                    'dft_density_surf2',
                    'dbcenter_surf2',
                    'dbfilling_surf2',
                    'dbwidth_surf2',
                    'dbskew_surf2',
                    'dbkurtosis_surf2',
                    'oxi_min_surf2',
                    'oxi_med_surf2',
                    'oxi_max_surf2',
                    'block_surf2',
                    'ne_outer_surf2',
                    'ne_s_surf2',
                    'ne_p_surf2',
                    'ne_d_surf2',
                    'ne_f_surf2',
                    'ionenergy_surf2',
                    'ground_state_magmom_surf2']
        else:
            ligand_atoms = atoms.info['ligand_atoms']
            numbers = atoms.numbers[ligand_atoms]
            # Import AtoML data on that element.
            extra_ads_params = ['atomic_radius', 'heat_of_formation',
                                'oxistates', 'block', 'econf', 'ionenergies']
            dat = list_mendeleev_params(numbers, params=default_params +
                                        extra_ads_params)
            result = list(np.nanmean(np.array(dat, dtype=float), axis=0))
            result += [np.nanmean([gs_magmom[z] for z in numbers])]
            return [len(ligand_atoms), len(np.unique(numbers))] + result

    def ads_nbonds(self, atoms=None):
        """ Function that takes an atoms object and returns a fingerprint
            vector with the number of C-H bonds and C-C bonds in the adsorbate.
            The adsorbate atoms must be specified in advance in
            atoms.info['ads_atoms']
        """
        if atoms is None:
            return ['nC-C', 'ndouble', 'nC-H', 'nO-H']
        else:
            ads_atoms = atoms[atoms.info['ads_atoms']]
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
        """ Function that takes an atoms objects and returns a fingerprint
            vector with averages of the atomic properties of the adsorbate.
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
            ads_atoms = atoms.info['ads_atoms']
            dat = []
            for a in ads_atoms:
                Z = atoms.numbers[a]
                ads_params = default_params + ['econf', 'ionenergies']
                mnlv = get_mendeleev_params(Z, params=ads_params)
                dat.append(mnlv[:-2] + list(n_outer(mnlv[-2])) +
                           [mnlv[-1]['1']])
            return list(np.nansum(dat, axis=0))

    def ads_av(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector with averages of the atomic properties of the adsorbate.
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
            ads_atoms = atoms.info['ads_atoms']
            dat = []
            for a in ads_atoms:
                Z = int(atoms.numbers[a])
                ads_params = default_params + ['econf', 'ionenergies']
                mnlv = get_mendeleev_params(Z, params=ads_params)
                dat.append(mnlv[:-2] + list(n_outer(mnlv[-2])) +
                           [mnlv[-1]['1']])
            return list(np.nanmean(dat, axis=0))

    def strain(self, atoms=None):
        if atoms is None:
            return ['strain_site', 'strain_term']
        else:
            chemi = atoms.numbers[atoms.info['chemisorbed_atoms']]
            bulk = atoms.info['key_value_pairs']['bulk']
            term = atoms.info['key_value_pairs']['term']
            bulkcomp = string2symbols(bulk)
            termcomp = string2symbols(term)
            rbulk = []
            rterm = []
            rsite = []
            for b in bulkcomp:
                rbulk.append(get_radius(atomic_numbers[b]))
            for t in termcomp:
                rterm.append(get_radius(atomic_numbers[t]))
            for z in chemi:
                rsite.append(get_radius(z))
            av_term = np.average(rterm)
            av_bulk = np.average(rbulk)
            av_site = np.average(rsite)
            strain_site = (av_site - av_bulk) / av_bulk
            strain_term = (av_term - av_bulk) / av_bulk
            return [strain_site, strain_term]

    def randomfpv(self, atoms=None):
        n = 20
        if atoms is None:
            return ['random'] * n
        else:
            return list(np.random.randint(0, 10, size=n))

    def delta_energy(self, atoms=None):
        if atoms is None:
            return ['Ef']
        else:
            try:
                delta = float(atoms.info['key_value_pair']['delta_energy'])
            except KeyError:
                delta = np.nan
            return [delta]

    def name(self, atoms=None):
        if atoms is None:
            return ['catapp_name']
        else:
            kvp = atoms.info['key_value_pairs']
            name = kvp['species'] + '*' + kvp['name'] + kvp['facet']
            return [name]

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
            Z_surf1_raw = [atoms.numbers[j] for j in atoms.info['ligand_atoms']]
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

    def get_dbid(self, atoms=None):
        if atoms is None:
            return ['dbid']
        else:
            return [int(atoms.info['dbid'])]

    def get_ctime(self, atoms=None):
        if atoms is None:
            return ['time_float']
        else:
            return [atoms.info['ctime']]
