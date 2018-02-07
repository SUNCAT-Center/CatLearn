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
from .periodic_table_data import (get_mendeleev_params,
                                  average_mendeleev_params,
                                  default_params)
from .database_adsorbate_api import layers_info, get_radius
from .neighbor_matrix import connection_matrix
import collections

block2number = {'s': 1,
                'p': 2,
                'd': 3,
                'f': 4}

# Text based feature.
facetdict = {'001': [1.], '0001step': [2.], '100': [3.],
             '110': [4.], '111': [5.], '211': [6.], '311': [7.],
             '532': [8.]}

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


def n_outer(econf):
    n_tot = 0
    n_s = 0
    n_p = 0
    n_d = 0
    n_f = 0
    for shell in econf.split(' ')[1:]:
        n_shell = 0
        if shell[-1].isalpha():
            n_shell = 1
        elif len(shell) == 3:
            n_shell = int(shell[-1])
        elif len(shell) == 4:
            n_shell = int(shell[-2:])
        n_tot += n_shell
        if 's' in shell:
            n_s += n_shell
        elif 'p' in shell:
            n_p += n_shell
        elif 'd' in shell:
            n_d += n_shell
        elif 'f' in shell:
            n_f += n_shell
    return n_tot, n_s, n_p, n_d, n_f


class AdsorbateFingerprintGenerator(object):
    def __init__(self):
        """ Class containing functions for fingerprint generation.
        """

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
                    'heat_of_formation_term',
                    'block_term',
                    'dft_bulk_modulus_term',
                    'dft_rhodensity_term',
                    'dbcenter_term',
                    'dbfilling_term',
                    'dbwidth_term',
                    'dbskew_term',
                    'dbkurtosis_term',
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
                term = atoms.info['terminaion']
            else:
                raise NotImplementedError("termination fingerprint.")
            # A = float(atoms.cell[0, 0]) * float(atoms.cell[1, 1])
            composition = [atomic_numbers[s] for s in string2symbols(term)]
            result = average_mendeleev_params(composition,
                                              params=default_params +
                                              default_extra_params)
            result += [np.mean([gs_magmom[atomic_numbers[s]] for s in
                                string2symbols(term)])]
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
            if ('key_value_pairs' in atoms.info and
                'bulk' in atoms.info['key_value_pairs']):
                    bulk = atoms.info['key_value_pairs']['bulk']
            elif 'bulk' in atoms.info:
                bulk = atoms.info['bulk']
            else:
                raise NotImplementedError("bulk fingerprint.")
            composition = [atomic_numbers[s] for s in string2symbols(bulk)]
            result = average_mendeleev_params(composition,
                                              params=default_params +
                                              default_extra_params)
            result += [np.mean([gs_magmom[atomic_numbers[s]] for s in
                                string2symbols(bulk)])]
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
                    'heat_of_formation_ads1',
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
            Z0 = int(atoms.info['Z_add1'])
            # Import AtoML data on that element.
            composition = [Z0]
            extra_params = ['heat_of_formation', 'block', 'econf',
                            'ionenergies']
            result = average_mendeleev_params(composition,
                                              params=default_params +
                                              extra_params)
            result += [gs_magmom[Z0]]
            return result

    def primary_surfatom(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector with properties of the surface metal atom closest to an add
            atom.
        """
        if atoms is None:
            return ['atomic_number_surf1',
                    'atomic_volume_surf1',
                    'boiling_point_surf1',
                    'density_surf1',
                    'dipole_polarizability_surf1',
                    'electron_affinity_surf1',
                    'group_id_surf1',
                    'lattice_constant_surf1',
                    'melting_point_surf1',
                    'period_surf1',
                    'vdw_radius_surf1',
                    'covalent_radius_cordero_surf1',
                    'en_allen_surf1',
                    'atomic_weight_surf1',
                    'heat_of_formation_surf1',
                    'dft_bulk_modulus_surf1',
                    'dft_rhodensity_surf1',
                    'dbcenter_surf1',
                    'dbfilling_surf1',
                    'dbwidth_surf1',
                    'dbskew_surf1',
                    'dbkurtosis_surf1',
                    'block_surf1',
                    'ne_outer_surf1',
                    'ne_s_surf1',
                    'ne_p_surf1',
                    'ne_d_surf1',
                    'ne_f_surf1',
                    'ionenergy_surf1',
                    'ground_state_magmom_surf1']
        else:
            composition = [atoms[j].number for j in atoms.info['i_surfnn']]
            result = average_mendeleev_params(composition,
                                              params=default_params +
                                              default_extra_params)
            result += [np.mean([gs_magmom[atoms[j].number] for j in
                                atoms.info['i_surfnn']])]
            return result

    def Z_add(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector containing the count of C, O, H and N atoms in the
            adsorbate.
        """
        if atoms is None:
            return ['total_num_C',
                    'total_num_O',
                    'total_num_H']
        else:
            nC = len([a.index for a in atoms if a.symbol == 'C'])
            nO = len([a.index for a in atoms if a.symbol == 'O'])
            # nN = len([a.index for a in atoms if a.symbol == 'N'])
            nH = len([a.index for a in atoms if a.symbol == 'H'])
            return [nC, nO, nH]  # , nN, nO]

    def primary_adds_nn(self, atoms=None, rtol=1.15):
        """ Function that takes an atoms objects and returns a fingerprint
            vector containing the count of C, O, H, N and also metal atoms,
            that are neighbors to the binding atom.
        """
        if atoms is None:
            return ['nn_num_C', 'nn_num_H', 'nn_num_M']
        else:
            # addsyms = ['H', 'C', 'O', 'N']
            # surf_atoms = atoms.info['surf_atoms']
            # ads_atoms = atoms.info['ads_atoms']
            # liste = []
            # for m in surf_atoms:
            #     for a in ads_atoms:
            #         d = atoms.get_distance(m, a, mic=True, vector=False)
            #         liste.append([a, d])
            primary_add = int(atoms.info['i_add1'])
            # Z_surf1 = int(atoms.info['Z_surf1'])
            Z_add1 = int(atoms.info['Z_add1'])
            dadd = get_radius(Z_add1)
            dH = get_radius(1)
            dC = get_radius(6)
            # dM = covalent_radii[Z_surf1]
            nH1 = len([a.index for a in atoms if a.symbol == 'H' and
                       atoms.get_distance(primary_add, a.index, mic=True) <
                       (dH+dadd)*rtol and a.index != primary_add])
            nC1 = len([a.index for a in atoms if a.symbol == 'C' and
                       atoms.get_distance(primary_add, a.index, mic=True) <
                       (dC+dadd)*rtol and a.index != primary_add])
            nM = len(atoms.info['i_surfnn'])
            # nM = len([a.index for a in atoms if a.symbol not in addsyms and
            #          atoms.get_distance(primary_add, a.index, mic=True) <
            #          (dM+dadd)*1.15])
            return [nC1, nH1, nM]  # , nN, nH]

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
            surf_atoms = atoms.info['surf_atoms']
            ads_atoms = atoms.info['ads_atoms']
            liste = []
            for m in surf_atoms:
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
                    'heat_of_formation_surf2',
                    'dft_bulk_modulus_surf2',
                    'dft_density_surf2',
                    'dbcenter_surf2',
                    'dbfilling_surf2',
                    'dbwidth_surf2',
                    'dbskew_surf2',
                    'dbkurtosis_surf2',
                    'block_surf2',
                    'ne_outer_surf2',
                    'ne_s_surf2',
                    'ne_p_surf2',
                    'ne_d_surf2',
                    'ne_f_surf2',
                    'ionenergy_surf2',
                    'ground_state_magmom_surf2']
        else:
            atoms.set_constraint()
            atoms = atoms.repeat([2, 2, 1])
            bulk_atoms, top_atoms = layers_info(atoms)
            symbols = atoms.get_chemical_symbols()
            numbers = atoms.numbers
            # Get a list of neighbors of binding surface atom(s).
            # Loop over binding surface atoms.
            ai = []
            adatoms = atoms.info['key_value_pairs']['species']
            for primary_surf in atoms.info['i_surfnn']:
                name = symbols[primary_surf]
                Z0 = numbers[primary_surf]
                r_bond = covalent_radii[Z0]
                # Loop over top atoms.
                for nni in top_atoms:
                    if nni == primary_surf or atoms[nni].symbol in adatoms:
                        continue
                    assert nni not in atoms.info['ads_atoms']
                    # Get pair distance and append to list.
                    d_nn = atoms.get_distance(primary_surf, nni, mic=True,
                                              vector=False)
                    ai.append([nni, d_nn])
            # mnlv0= get_mendeleev_params(Z0,extra_params=['heat_of_formation',
            #                                               'dft_bulk_modulus',
            #                                               'dft_density',
            #                                               'dbcenter',
            #                                               'dbfilling',
            #                                               'dbwidth',
            #                                               'dbskew',
            #                                               'dbkurt',
            #                                               'block',
            #                                               'econf',
            #                                               'ionenergies'])
            dat = []  # [mnlv0[:-3] + [float(block2number[mnlv0[-3]])] +
            #       list(n_outer(mnlv0[-2])) + [mnlv0[-1]['1']] +
            #       [float(ground_state_magnetic_moments[Z0])]]
            n = 0
            q_self = []
            for nn in ai:
                q = nn[0]
                Znn = int(numbers[q])
                r_bond_nn = covalent_radii[Znn]
                if q != primary_surf and nn[1] < rtol * (r_bond_nn+r_bond):
                    sym = symbols[q]
                    mnlv = get_mendeleev_params(Znn, extra_params=[
                                                 'heat_of_formation',
                                                 'dft_bulk_modulus',
                                                 'dft_density',
                                                 'dbcenter',
                                                 'dbfilling',
                                                 'dbwidth',
                                                 'dbskew',
                                                 'dbkurt',
                                                 'block',
                                                 'econf',
                                                 'ionenergies'])
                    n += 1
                    dat.append(mnlv[:-3] + [float(block2number[mnlv[-3]])] +
                               list(n_outer(mnlv[-2])) + [mnlv[-1]['1']] +
                               [float(gs_magmom[Znn])])
                    if sym == name:
                        q_self.append(q)
            n_self = len(q_self)
            if len(dat) == 0:
                raise ValueError("atoms.info['i_surf'] is not correct" +
                                 " for dbid " + str(atoms.info['dbid']))
            return [n, n_self] + list(np.nanmean(dat, axis=0, dtype=float))

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
                mnlv = get_mendeleev_params(Z, extra_params=['econf',
                                                             'ionenergies'])
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
                mnlv = get_mendeleev_params(Z, extra_params=['econf',
                                                             'ionenergies'])
                dat.append(mnlv[:-2] + list(n_outer(mnlv[-2])) +
                           [mnlv[-1]['1']])
            return list(np.nanmean(dat, axis=0))

    def strain(self, atoms=None):
        if atoms is None:
            return ['strain_surf1', 'strain_term']
        else:
            Z_add1 = int(atoms.info['Z_add1'])
            bulk = atoms.info['key_value_pairs']['bulk']
            term = atoms.info['key_value_pairs']['term']
            bulkcomp = string2symbols(bulk)
            termcomp = string2symbols(term)
            rbulk = []
            rterm = []
            for b in bulkcomp:
                rbulk.append(covalent_radii[atomic_numbers[b]])
            for t in termcomp:
                rterm.append(covalent_radii[atomic_numbers[t]])
            av_term = np.average(rterm)
            av_bulk = np.average(rbulk)
            strain1 = (covalent_radii[Z_add1] - av_bulk) / av_bulk
            strain_term = (av_term - av_bulk) / av_bulk
            return [strain1, strain_term]

    def randomfpv(self, atoms=None):
        n = 20
        if atoms is None:
            return ['random']*n
        else:
            return list(np.random.randint(0, 10, size=n))

    def info2Ef(self, atoms=None):
        if atoms is None:
            return ['Ef']
        else:
            return [float(atoms.info['Ef'])]

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
            Z_surf1_raw = [atoms.numbers[j] for j in atoms.info['i_surfnn']]
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
                raise AssertionError("Tertiary surfaces not supported.")
            f1 = get_mendeleev_params(z1,
                                      extra_params=['heat_of_formation',
                                                    #'dft_bulk_modulus',
                                                    #'dft_density',
                                                    #'dbcenter',
                                                    #'dbfilling',
                                                    #'dbwidth',
                                                    #'dbskew',
                                                    #'dbkurt',
                                                    'block',
                                                    'econf',
                                                    'ionenergies'])
            f1 = f1[:-3] + [float(block2number[f1[-3]])] + \
                list(n_outer(f1[-2])) + [f1[-1]['1']] + \
                [float(gs_magmom[z1])]
            if z1 == z2:
                f2 = f1
            else:
                f2 = get_mendeleev_params(z2,
                                          extra_params=['heat_of_formation',
                                                        #'dft_bulk_modulus',
                                                        #'dft_density',
                                                        #'dbcenter',
                                                        #'dbfilling',
                                                        #'dbwidth',
                                                        #'dbskew',
                                                        #'dbkurt',
                                                        'block',
                                                        'econf',
                                                        'ionenergies'])
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
            return ['ctime']
        else:
            return [int(atoms.info['ctime'])]
