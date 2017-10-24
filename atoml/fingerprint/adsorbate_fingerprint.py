# -*- coding: utf-8 -*-
"""
Slab adsorbate fingerprint functions for machine learning

Created on Tue Dec  6 14:09:29 2016

@author: mhangaard

"""
from __future__ import print_function

import numpy as np
from ase.atoms import string2symbols
from ase.data import ground_state_magnetic_moments, covalent_radii
from ase.data import atomic_numbers
from .periodic_table_data import get_mendeleev_params
from .db2thermo import layers_info
from .neighbor_matrix import connection_matrix


block2number = {'s': 1,
                'p': 2,
                'd': 3,
                'f': 4}


def n_outer(econf):
    n_tot = 0
    ns = 0
    np = 0
    nd = 0
    nf = 0
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
            ns += n_shell
        elif 'p' in shell:
            np += n_shell
        elif 'd' in shell:
            nd += n_shell
        elif 'f' in shell:
            nf += n_shell
    return n_tot, ns, np, nd, nf


def info2primary_index(atoms):
    liste = []
    surf_atoms = atoms.info['surf_atoms']
    add_atoms = atoms.info['add_atoms']
    for m in surf_atoms:
        for a in add_atoms:
            d = atoms.get_distance(m, a, mic=True, vector=False)
            liste.append([a, m, d])
    L = np.array(liste)
    i = np.argmin(L[:, 2])
    i_add1 = L[i, 0]
    i_surf1 = L[i, 1]
    Z_add1 = atoms.numbers[int(i_add1)]
    Z_surf1 = atoms.numbers[int(i_surf1)]
    return i_add1, i_surf1, Z_add1, Z_surf1


class AdsorbateFingerprintGenerator(object):
    def __init__(self):
        """ Class containing functions for fingerprint generation.
        """

    def term(self, atoms=None):  # , adds_dict):
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
            name = atoms.info['key_value_pairs']['term']
            # A = float(atoms.cell[0, 0]) * float(atoms.cell[1, 1])
            comp = string2symbols(name)
            dat = []
            # np.unique could be used.
            for symb in comp:
                Z = atomic_numbers[symb]
                mnlv = get_mendeleev_params(Z,
                                            extra_params=['heat_of_formation',
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
                dat.append(mnlv[:-3] + [float(block2number[mnlv[-3]])] +
                           list(n_outer(mnlv[-2])) +
                           [mnlv[-1]['1']] +
                           [float(ground_state_magnetic_moments[Z])])
            return list(np.nanmean(dat, axis=0, dtype=float))

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
            name = atoms.info['key_value_pairs']['bulk']
            bulkcomp = string2symbols(name)
            dat = []
            # np.unique could be used.
            for symb in bulkcomp:
                Z = atomic_numbers[symb]
                mnlv = get_mendeleev_params(Z,
                                            extra_params=['heat_of_formation',
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
                dat.append(mnlv[:-3] + [float(block2number[mnlv[-3]])] +
                           list(n_outer(mnlv[-2])) +
                           [mnlv[-1]['1']] +
                           [float(ground_state_magnetic_moments[Z])])
            return list(np.nanmean(dat, axis=0, dtype=float))

    def primary_addatom(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector containing properties of the closest add atom to a surface
            metal atom.
        """
        if atoms is None:
            return ['atomic_number_add1',
                    'atomic_volume_add1',
                    'boiling_point_add1',
                    'density_add1',
                    'dipole_polarizability_add1',
                    'electron_affinity_add1',
                    'group_id_add1',
                    'lattice_constant_add1',
                    'melting_point_add1',
                    'period_add1',
                    'vdw_radius_add1',
                    'covalent_radius_cordero_add1',
                    'en_allen_add1',
                    'atomic_weight_add1',
                    'heat_of_formation_add1',
                    'block_add1',
                    'ne_outer_add1',
                    'ne_s_add1',
                    'ne_p_add1',
                    'ne_d_add1',
                    'ne_f_add1',
                    'ionenergy_add1',
                    'ground_state_magmom_add1']
        else:
            # Get atomic number of alpha adsorbate atom.
            Z0 = int(atoms.info['Z_add1'])
            # Import AtoML data on that element.
            dat = get_mendeleev_params(Z0, extra_params=['heat_of_formation',
                                                         'block',
                                                         'econf',
                                                         'ionenergies'])
            result = dat[:-3] + [block2number[dat[-3]]] + \
                list(n_outer(dat[-2])) + [dat[-1]['1']]
            # Append ASE data on that element.
            result += [float(ground_state_magnetic_moments[Z0])]
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
                    #'dbkurtosis_surf1',
                    'block_surf1',
                    'ne_outer_surf1',
                    'ne_s_surf1',
                    'ne_p_surf1',
                    'ne_d_surf1',
                    'ne_f_surf1',
                    'ionenergy_surf1',
                    'strain_surf1',
                    'ground_state_magmom_surf1']
        else:
            bulk = atoms.info['key_value_pairs']['bulk']
            bulkcomp = string2symbols(bulk)
            r = []
            for bulk in bulkcomp:
                r.append(covalent_radii[atomic_numbers[bulk]])
            r0 = np.average(r)
            # Z = int(atoms.info['Z_surf1'])
            dat = []
            for j in atoms.info['i_surfnn']:
                Z = int(atoms[j].number)
                mnlv = get_mendeleev_params(Z,
                                            extra_params=['heat_of_formation',
                                                          'dft_bulk_modulus',
                                                          'dft_density',
                                                          'dbcenter',
                                                          'dbfilling',
                                                          'dbwidth',
                                                          'dbskew',
                                                          #'dbkurt',
                                                          'block',
                                                          'econf',
                                                          'ionenergies'])
                strain = (covalent_radii[Z] - r0) / r0
                dat.append(mnlv[:-3] + [float(block2number[mnlv[-3]])] +
                           list(n_outer(mnlv[-2])) +
                           [mnlv[-1]['1']] + [strain] +
                           [float(ground_state_magnetic_moments[Z])])
                return list(np.nanmean(dat, axis=0, dtype=float))

    def Z_add(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector containing the count of C, O, H and N atoms in the
            adsorbate.
        """
        if atoms is None:
            return ['total_num_C',
                    'total_num_O' ,# 'total_num_N',
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
            # add_atoms = atoms.info['add_atoms']
            # liste = []
            # for m in surf_atoms:
            #     for a in add_atoms:
            #         d = atoms.get_distance(m, a, mic=True, vector=False)
            #         liste.append([a, d])
            primary_add = int(atoms.info['i_add1'])
            # Z_surf1 = int(atoms.info['Z_surf1'])
            Z_add1 = int(atoms.info['Z_add1'])
            dH = covalent_radii[1]
            dC = covalent_radii[6]
            # dM = covalent_radii[Z_surf1]
            dadd = covalent_radii[Z_add1]
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
            add_atoms = atoms.info['add_atoms']
            liste = []
            for m in surf_atoms:
                for a in add_atoms:
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
            nM = len([a.index for a in atoms if a.symbol not in add_atoms and
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
            atoms = atoms.repeat([2, 2, 1])
            bulk_atoms, top_atoms = layers_info(atoms)
            symbols = atoms.get_chemical_symbols()
            numbers = atoms.numbers
            # Get a list of neighbors of binding surface atom(s).
            # Loop over binding surface atoms.
            ai = []
            adatoms = ['H', 'C', 'N', 'O']
            for primary_surf in atoms.info['i_surfnn']:
                name = symbols[primary_surf]
                Z0 = numbers[primary_surf]
                r_bond = covalent_radii[Z0]
                # Loop over top atoms.
                for nni in top_atoms:
                    if nni == primary_surf or atoms[nni].symbol in adatoms:
                        continue
                    assert nni not in atoms.info['add_atoms']
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
                Znn = numbers[q]
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
                               [float(ground_state_magnetic_moments[Znn])])
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
            atoms.info['add_atoms']
        """
        if atoms is None:
            return ['nC-C', 'ndouble', 'nC-H', 'nO-H']
        else:
            add_atoms = atoms[atoms.info['add_atoms']]
            A = connection_matrix(add_atoms, periodic=True, dx=0.2)
            Hindex = [a.index for a in add_atoms if a.symbol == 'H']
            Cindex = [a.index for a in add_atoms if a.symbol == 'C']
            Oindex = [a.index for a in add_atoms if a.symbol == 'O']
            nCC = 0
            nCH = 0
            nC2 = 0
            nOH = 0
            for o in Oindex:
                nOH += np.sum(A[Hindex, o])
            for a in Cindex:
                nCC += np.sum(A[Cindex, a])
                nCH += np.sum(A[Hindex, a])
                nC2 += 4 - (nCC + nCH)
            return [nCC, nC2, nCH, nOH]

    def adds_av(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector with averages of the atomic properties of the adsorbate.
        """
        if atoms is None:
            return ['atomic_number_add_av',
                    'atomic_volume_add_av',
                    'boiling_point_add_av',
                    'density_add_av',
                    'dipole_polarizability_add_av',
                    'electron_affinity_add_av',
                    'group_id_add_av',
                    'lattice_constant_add_av',
                    'melting_point_add_av',
                    'period_add_av',
                    'vdw_radius_add_av',
                    'covalent_radius_cordero_add_av',
                    'en_allen_add_av',
                    'atomic_weight_add_av',
                    'ne_outer_add_av',
                    'ne_s_add_av',
                    'ne_p_add_av',
                    'ne_d_add_av',
                    'ne_f_add_av',
                    'ionenergy_av']
        else:
            add_atoms = atoms.info['add_atoms']
            dat = []
            for a in add_atoms:
                Z = atoms.numbers[a]
                mnlv = get_mendeleev_params(Z, extra_params=['econf',
                                                             'ionenergies'])
                dat.append(mnlv[:-2] + list(n_outer(mnlv[-2])) +
                           [mnlv[-1]['1']])
            return list(np.average(dat, axis=0))

    def randomfpv(self, atoms=None):
        n = 20
        if atoms is None:
            return ['random']*n
        else:
            return list(np.random.randint(0, 10, size=n))

    def info2Ef(self, atoms):
        if atoms is None:
            ['Ef']
        else:
            return [float(atoms.info['Ef'])]

    def get_dbid(self, atoms=None):
        if atoms is None:
            return ['dbid']
        else:
            return [int(atoms.info['dbid'])]

    def get_keyvaluepair(self, atoms=None, field_name='None'):
        if atoms is None:
            return ['kvp_'+field_name]
        else:
            field_value = float(atoms.info['key_value_pairs'][field_name])
            [field_value]
            return
