# -*- coding: utf-8 -*-
"""
Slab adsorbate fingerprint functions for machine learning

Created on Tue Dec  6 14:09:29 2016

@author: mhangaard

"""
from __future__ import print_function

import warnings
import numpy as np

import ase.db
from ase.atoms import string2symbols
from ase.data import ground_state_magnetic_moments, covalent_radii
from ase.data import chemical_symbols

try:
    from mendeleev import element
except ImportError:
    print('mendeleev not imported')

def n_outer(mnlv):
    econf = mnlv.econf.split(' ')[1:]
    n_tot = 0
    ns = 0
    np = 0
    nd = 0
    nf = 0
    for shell in econf:
        n_shell = 0
        if shell[-1].isalpha:
            n_shell = 1
        else:
            n_shell = int(shell[-1])
        n_tot += n_shell
        if 's' in shell:
            ns += n_shell
        if 'p' in shell:
            ns += n_shell
        if 'd' in shell:
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
    def __init__(self, bulkdb=None):
        """
        Input:
            moldb:  path/filename.db
            bulkdb: path/filename.db
            slabs:  path/filename.db
            parameters: dict with keywords:
                            vacuum,
                            layers,
                            PW,
                            kpts,
                            PSP
        """
        # If bulks are supplied, add their calculated properties to dicts.
        if bulkdb is not None:
            self.bulks = bulkdb
            [self.rho, self.Z, self.eos_B,
             self.dbcenter,
             self.dbfilling,
             self.dbwidth,
             self.dbskew,
             self.dbkurt,
             self.d_atoms] = self.get_bulk()

    def get_bulk(self):
        """ Returns dictionaries with properties calculated for bulk
        structures contained in self.bulks.
        """
        c = ase.db.connect(self.bulks)
        s = c.select()
        Z = {}
        rho = {}
        eos_B = {}
        d_atoms = {}
        dbcenter = {}
        dbfilling = {}
        dbwidth = {}
        dbskew = {}
        dbkurt = {}
        for d in s:
            name = str(d.name)
            Z[name] = int(d.numbers[0])
            rho[name] = len(d.numbers) / float(d.volume)
            eos_B[name] = float(d.eos_B)
            atoms = c.get_atoms(int(d.id))
            d_atoms[name] = (atoms.get_distances(0, range(1, len(atoms)),
                                                 mic=True)).min()
            if 'dbcenter' in d.key_value_pairs:
                dbcenter[name] = float(d.dbcenter)
                dbfilling[name] = float(d.dbfilling)
                dbwidth[name] = float(d.dbwidth)
                dbskew[name] = float(d.dbskew)
                dbkurt[name] = float(d.dbkurt)
        return (rho, Z, eos_B, dbcenter, dbfilling, dbwidth, dbskew, dbkurt,
                d_atoms)

    def db2atoms_fps(self):  # , adds_dict):
        c = ase.db.connect(self.slabs)
        s = c.select(['series!=slab', 'C=0', 'layers=5', 'facet=1x1x1',
                      'kpts=4x6', 'PW=500', 'PSP=gbrv1.5pbe'])
        trajs = []
        for d in s:
            dbid = int(d.id)
            name = str(d.name)
            atoms = c.get_atoms(dbid)
            atoms.info['key_value_pairs'] = {}  # d.key_value_pairs
            atoms.info['key_value_pairs']['Z'] = self.Z[name]
            atoms.info['key_value_pairs']['rho_bulk'] = self.rho[name]
            A = float(d.cell[0, 0]) * float(d.cell[1, 1])
            atoms.info['key_value_pairs']['rho_surf'] = len(d.numbers) / A
            atoms.info['key_value_pairs']['Ef'] = (self.Ef[str(d.series) +
                                                           '_' + name + '_' +
                                                           str(d.phase) + '_' +
                                                           str(d.facet)])
            trajs.append(atoms)
        return trajs

    def term(self, atoms=None):  # , adds_dict):
        """ Returns a fingerprint vector with propeties of the element name
        saved in the atoms.info['key_value_pairs']['term'] """
        if atoms is None:
            return ['rho_vol_term', 'rho_A_term', 'B_eos_term',
                    'dbcenter_term',
                    'dbfilling_term',
                    'dbwidth_term',
                    'dbskew_term',
                    'dbkurtosis_term']
        else:
            # series = atoms.info['key_value_pairs']['series']
            name = atoms.info['key_value_pairs']['term']
            # phase = atoms.info['key_value_pairs']['phase']
            # facet = atoms.info['key_value_pairs']['facet']
            # key = series+'_'+name+'_'+phase+'_'+facet
            A = float(atoms.cell[0, 0]) * float(atoms.cell[1, 1])
            try:
                dbcenter = float(self.dbcenter[name])
                dbfilling = float(self.dbfilling[name])
                dbwidth = float(self.dbwidth[name])
                dbskew = float(self.dbskew[name])
                dbkurt = float(self.dbkurt[name])
            except KeyError:
                dbcenter = np.NaN
                dbfilling = np.NaN
                dbwidth = np.NaN
                dbskew = np.NaN
                dbkurt = np.NaN
                warnings.warn(name+' has no d-band info.')
            return [
                float(self.rho[name]),
                len(atoms.numbers) / A,
                float(self.eos_B[name]),
                dbcenter,
                dbfilling,
                dbwidth,
                dbskew,
                dbkurt,
                ]

    def bulk(self, atoms=None):  # , adds_dict):
        """ Returns a fingerprint vector with propeties of the element name
        saved in the atoms.info['key_value_pairs']['bulk'] """
        if atoms is None:
            return ['rho_vol_bulk', 'rho_A_bulk', 'B_eos_bulk',
                    'dbcenter_bulk',
                    'dbfilling_bulk',
                    'dbwidth_bulk',
                    'dbskew_bulk',
                    'dbkurtosis_bulk']
        else:
            # series = atoms.info['key_value_pairs']['series']
            name = atoms.info['key_value_pairs']['bulk']
            # phase = atoms.info['key_value_pairs']['phase']
            # facet = atoms.info['key_value_pairs']['facet']
            # key = series+'_'+name+'_'+phase+'_'+facet
            A = float(atoms.cell[0, 0]) * float(atoms.cell[1, 1])
            try:
                dbcenter = float(self.dbcenter[name])
                dbfilling = float(self.dbfilling[name])
                dbwidth = float(self.dbwidth[name])
                dbskew = float(self.dbskew[name])
                dbkurt = float(self.dbkurt[name])
            except KeyError:
                dbcenter = np.NaN
                dbfilling = np.NaN
                dbwidth = np.NaN
                dbskew = np.NaN
                dbkurt = np.NaN
                warnings.warn(name+' has no d-band info.')
            return [
                float(self.rho[name]),
                len(atoms.numbers) / A,
                float(self.eos_B[name]),
                dbcenter,
                dbfilling,
                dbwidth,
                dbskew,
                dbkurt,
                ]

    def primary_addatom(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector containing properties of the closest add atom to a surface
            metal atom.
        """
        if atoms is None:
            return ['Z_add1', 'period_add1', 'group_id_add1',  # 'en_allen',
                    'electron_affinity_add1', 'dipole_polarizability_add1',
                    'en_pauling_add1', 'atomic_radius_add1', 'vdw_radius_add1',
                    'ion_e_add1', 'ground_state_magmom_add1']
        else:
            Z0 = int(atoms.info['Z_add1'])
            mnlv = element(Z0)
            result = [
                 Z0,
                 int(mnlv.period),
                 float(mnlv.group_id),
                 float(mnlv.electron_affinity),
                 float(mnlv.dipole_polarizability),
                 # float(mnlv.en_allen),
                 float(mnlv.en_pauling),
                 float(mnlv.atomic_radius),
                 float(mnlv.vdw_radius),
                 float(mnlv.ionenergies[1]),
                 float(ground_state_magnetic_moments[Z0])
                 ]
            return result

    def primary_surfatom(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector with properties of the surface metal atom closest to an add
            atom.
        """
        if atoms is None:
            return ['Z', 'period_surf1', 'group_id_surf1',
                    'electron_affinity_surf1',
                    'dipole_polarizability_surf1',
                    'heat_of_formationsurf1',
                    'melting_point_surf1',
                    'boiling_point_surf1',
                    # 'thermal_conductivity_surf1',
                    # 'specific_heat_surf1',
                    'en_allen_surf1',
                    'en_pauling_surf1', 'atomic_radius_surf1',
                    'vdw_radius_surf1', 'ion_e_surf1',
                    'dbcenter_surf1',
                    'dbfilling_surf1',
                    'dbwidth_surf1',
                    'dbskew_surf1',
                    'dbkurtosis_surf1',
                    'ne_outer_surf1',
                    'ne_s_surf1',
                    'ne_p_surf1',
                    'ne_d_surf1',
                    'ne_f_surf1',
                    'ground_state_magmom_surf1']
        else:
            Z0 = int(atoms.info['Z_surf1'])
            mnlv = element(Z0)
            name = chemical_symbols[Z0]
            try:
                dbcenter = float(self.dbcenter[name])
                dbfilling = float(self.dbfilling[name])
                dbwidth = float(self.dbwidth[name])
                dbskew = float(self.dbskew[name])
                dbkurt = float(self.dbkurt[name])
            except KeyError:
                dbcenter = np.NaN
                dbfilling = np.NaN
                dbwidth = np.NaN
                dbskew = np.NaN
                dbkurt = np.NaN
                print(name+' has no d-band info.')
            n_tot, n_s, n_p, n_d, n_f = n_outer(mnlv)
            return [Z0,
                    int(mnlv.period),
                    int(mnlv.group_id),
                    float(mnlv.electron_affinity),
                    float(mnlv.dipole_polarizability),
                    float(mnlv.heat_of_formation),
                    float(mnlv.melting_point),
                    float(mnlv.boiling_point),
                    # float(mnlv.thermal_conductivity),
                    # float(mnlv.specific_heat),
                    float(mnlv.en_allen),
                    float(mnlv.en_pauling),
                    float(mnlv.atomic_radius),
                    float(mnlv.vdw_radius),
                    float(mnlv.ionenergies[1]),
                    dbcenter,
                    dbfilling,
                    dbwidth,
                    dbskew,
                    dbkurt,
                    n_tot,
                    n_s,
                    n_p,
                    n_d,
                    n_f,
                    float(ground_state_magnetic_moments[Z0])
                    ]

    def Z_add(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector containing the count of C, O, H and N atoms in the
            adsorbate.
        """
        if atoms is None:
            return ['total_num_C', 'total_num_O', 'total_num_N', 'total_num_H']
        else:
            nC = len([a.index for a in atoms if a.symbol == 'C'])
            nO = len([a.index for a in atoms if a.symbol == 'O'])
            nN = len([a.index for a in atoms if a.symbol == 'N'])
            nH = len([a.index for a in atoms if a.symbol == 'H'])
            return [nC, nH, nN, nO]

    def primary_adds_nn(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector containing the count of C, O, H, N and also metal atoms,
            that are neighbors to the binding atom.
        """
        if atoms is None:
            return ['nn_num_C', 'nn_num_H', 'nn_num_M']
        else:
            addsyms = ['H', 'C', 'O', 'N']
            # surf_atoms = atoms.info['surf_atoms']
            # add_atoms = atoms.info['add_atoms']
            # liste = []
            # for m in surf_atoms:
            #     for a in add_atoms:
            #         d = atoms.get_distance(m, a, mic=True, vector=False)
            #         liste.append([a, d])
            primary_add = int(atoms.info['i_add1'])
            Z_surf1 = int(atoms.info['Z_surf1'])
            Z_add1 = int(atoms.info['Z_add1'])
            dM = covalent_radii[Z_surf1]
            dadd = covalent_radii[Z_add1]
            nH1 = len([a.index for a in atoms if a.symbol == 'H' and
                       atoms.get_distance(primary_add, a.index, mic=True) <
                       1.3 and a.index != primary_add])
            nC1 = len([a.index for a in atoms if a.symbol == 'C' and
                       atoms.get_distance(primary_add, a.index, mic=True) <
                       1.3 and a.index != primary_add])
            nM = len([a.index for a in atoms if a.symbol not in addsyms and
                      atoms.get_distance(primary_add, a.index, mic=True) <
                      (dM+dadd)*1.3])
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
            add_atoms = ['add_atoms']
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
                       1.3 and a.index != secondary_add])
            nC1 = len([a.index for a in atoms if a.symbol == 'C' and
                      atoms.get_distance(secondary_add, a.index, mic=True) <
                      1.3 and a.index != secondary_add])
            nM = len([a.index for a in atoms if a.symbol not in add_atoms and
                     atoms.get_distance(secondary_add, a.index, mic=True) <
                     2.35])
            return [nC1, nH1, nM]  # , nN, nH]

    def primary_surf_nn(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector containing the count of nearest neighbors and properties of
            the nearest neighbors.
        """
        if atoms is None:
            return ['num_nn', 'nn_num_identical',
                    'av_dbcenter',
                    'av_dbfilling',
                    'av_dbwidth',
                    'av_dbskew',
                    'av_dbkurtosis',
                    'av_heat_of_formation',
                    'av_melting_point',
                    'av_boiling_point',
                    # 'av_specific_heat',
                    'av_en_allen',
                    'av_en_pauling',
                    'av_ionenergies',
                    'av_atomic_radius',
                    'av_ground_state_magmom',
                    'av_ne_outer',
                    'av_ne_s',
                    'av_ne_p',
                    'av_ne_d',
                    'av_ne_f',
                    ]
        else:
            add_atoms = atoms.info['add_atoms']
            atoms = atoms.repeat([1, 2, 1])
            surf_atoms = [a.index for a in atoms if a.index not in add_atoms]
            liste = []
            for m in surf_atoms:
                for a in add_atoms:
                    d = atoms.get_distance(m, a, mic=True, vector=False)
                    liste.append([m, d])
            L = np.array(liste)
            i = np.argmin(L[:, 1])
            primary_surf = int(L[i, 0])
            symbols = atoms.get_chemical_symbols()
            numbers = atoms.numbers
            name = symbols[primary_surf]
            Z0 = numbers[primary_surf]
            r_bond = covalent_radii[Z0]
            ai = []
            for nni in surf_atoms:
                d_nn = atoms.get_distance(primary_surf, nni, mic=True,
                                          vector=False)
                ai.append([nni, d_nn])
            # ai = np.where(atoms.get_distances(primary_surf, surf_atoms,
            #                                  mic=True) < r_bond)
            try:
                dbcenter = [float(self.dbcenter[name])]
                dbfilling = [float(self.dbfilling[name])]
                dbwidth = [float(self.dbwidth[name])]
                dbskew = [float(self.dbskew[name])]
                dbkurt = [float(self.dbkurt[name])]
            except KeyError:
                dbcenter = []
                dbfilling = []
                dbwidth = []
                dbskew = []
                dbkurt = []
            mnlv0 = element(name)
            heat_of_formation = [mnlv0.heat_of_formation]
            melting_point = [mnlv0.melting_point]
            boiling_point = [mnlv0.boiling_point]
            # specific_heat=[mnlv0.specific_heat]
            en_allen = [mnlv0.en_allen]
            en_pauling = [mnlv0.en_pauling]
            ionenergies = [mnlv0.ionenergies[1]]
            atomic_radius = [mnlv0.atomic_radius]
            ground_state_magmom = [ground_state_magnetic_moments[Z0]]
            n_tot0, n_s0, n_p0, n_d0, n_f0 = n_outer(mnlv0)
            n_tot = [n_tot0]
            n_s = [n_s0]
            n_p = [n_p0]
            n_d = [n_d0]
            n_f = [n_f0]
            q_self = []
            n = 0
            for nn in ai:
                q = nn[0]
                Znn = numbers[q]
                r_bond_nn = covalent_radii[Znn]
                if q != primary_surf and 1.2*(r_bond_nn+r_bond) > nn[1]:
                    sym = symbols[q]
                    mnlv = element(sym)
                    n += 1
                    if sym in self.dbcenter:
                        dbcenter.append(self.dbcenter[sym])
                        dbfilling.append(self.dbfilling[sym])
                        dbwidth.append(self.dbwidth[sym])
                        dbskew.append(self.dbskew[sym])
                        dbkurt.append(self.dbkurt[sym])
                    heat_of_formation.append(mnlv.heat_of_formation)
                    # specific_heat.append(mnlv.specific_heat)
                    en_allen.append(mnlv.en_allen)
                    en_pauling.append(mnlv.en_pauling)
                    ionenergies.append(mnlv.ionenergies[1])
                    atomic_radius.append(r_bond_nn)
                    ground_state_magmom.append(
                        ground_state_magnetic_moments[numbers[q]])
                    n_tot0, n_s0, n_p0, n_d0, n_f0 = n_outer(mnlv)
                    n_tot.append(n_tot0)
                    n_s.append(n_s0)
                    n_p.append(n_d0)
                    n_d.append(n_p0)
                    n_f.append(n_f0)
                    if sym == name:
                        q_self.append(q)
            av_dbcenter = np.average(dbcenter)
            av_dbfilling = np.average(dbfilling)
            av_dbwidth = np.average(dbwidth)
            av_dbskew = np.average(dbskew)
            av_dbkurt = np.average(dbkurt)
            av_heat_of_formation = np.average(heat_of_formation)
            av_melting_point = np.average(melting_point)
            av_boiling_point = np.average(boiling_point)
            # av_specific_heat = np.average(specific_heat)
            av_en_allen = np.average(en_allen)
            av_en_pauling = np.average(en_pauling)
            av_ionenergies = np.average(ionenergies)
            av_atomic_radius = np.average(atomic_radius)
            av_ground_state_magmom = np.average(ground_state_magmom)
            av_ne_outer = np.average(n_tot)
            av_n_s = np.average(n_s)
            av_n_p = np.average(n_p)
            av_n_d = np.average(n_d)
            av_n_f = np.average(n_f)
            n_self = len(q_self)
            return [n, n_self,
                    av_dbcenter,
                    av_dbfilling,
                    av_dbwidth,
                    av_dbskew,
                    av_dbkurt,
                    av_heat_of_formation,
                    av_melting_point,
                    av_boiling_point,
                    # av_specific_heat,
                    av_en_allen,
                    av_en_pauling,
                    av_ionenergies,
                    av_atomic_radius,
                    av_ground_state_magmom,
                    av_ne_outer,
                    av_n_s,
                    av_n_p,
                    av_n_d,
                    av_n_f,
                    ]

    def adds_sum(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector containing sums of the atomic properties of the adsorbate.
        """
        if atoms is None:
            return ['sum_electron_affinity', 'av_electron_affinity',
                    'sum_en_allen', 'av_en_allen', 'sum_en_pauling',
                    'av_en_pauling']
        else:
            add_atoms = atoms.info['add_atoms']
            electron_affinity = 0
            en_allen = 0
            en_pauling = 0
            heat_of_formation = 0
            L = len(add_atoms)
            for a in add_atoms:
                Z0 = atoms.numbers[a]
                electron_affinity += float(element(Z0).electron_affinity)
                en_allen += float(element(Z0).en_allen)
                en_pauling += float(element(Z0).en_pauling)
                heat_of_formation += float(element(Z0).heat_of_formation)
            result = [electron_affinity, electron_affinity/L, en_allen,
                      en_allen/L, en_pauling, en_pauling/L]
            return result

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

    def get_dbid(self, atoms):
        if atoms is None:
            ['dbid']
        else:
            return int(atoms.info['dbid'])

    def get_keyvaluepair(self, atoms=None, field_name='None'):
        if atoms is None:
            return ['kvp_'+field_name]
        else:
            field_value = float(atoms['key_value_pairs'][field_name])
            [field_value]
            return
