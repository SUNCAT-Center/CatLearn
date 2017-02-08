# -*- coding: utf-8 -*-
"""
Slab adsorbate fingerprint functions for machine learning

Created on Tue Dec  6 14:09:29 2016

@author: mhangaard

"""

import warnings
import numpy as np
import ase.db
from ase.atoms import string2symbols
from atoml.db2thermo import db2mol, db2surf, mol2ref, get_refs  # get_formation_energies
from mendeleev import element
from random import random


class AdsorbateFingerprintGenerator(object):
    def __init__(self, moldb='mol.db', bulkdb=None, slabs=None):
        self.mols = moldb
        parameters = {'vacuum': 8, 'layers': 5, 'PW': 500, 'kpts': '4x6',
                      'PSP': 'gbrv1.5pbe'}
        if bulkdb is not None:
            self.bulks = bulkdb
            self.rho, self.Z, self.eos_B, self.dbcenter, self.dbfilling, self.d_atoms = self.get_bulk()
        abinitio_energies, dbids = db2mol(moldb, ['vacuum='+str(parameters['vacuum']),'PW='+str(parameters['PW'])])
        self.mol_dict = mol2ref(abinitio_energies)
        self.slabs = slabs
        if slabs is not None:
            surf_energies, surf_dbids = db2surf(slabs, ['series=slab','layers='+str(parameters['layers']),'kpts='+str(parameters['kpts']),'PW='+str(parameters['PW']),'PSP='+str(parameters['PSP'])])
            abinitio_energies.update(surf_energies)
            dbids.update(surf_dbids)
            # surf_energies211, surf_frequencies211, surf_contribs211, surf_dbids211 = db2surf(slabs, ['Al=0','Rh=0','Pt=0','layers=4','facet=2x1x1','kpts=4x4','PW=500','PSP=gbrv1.5pbe'])
            # abinitio_energies.update(surf_energies211)
            # frequency_dict.update(surf_frequencies211)
            # contribs.update(surf_contribs211)
            # dbids.update(surf_dbids211)
            self.ref_dict = get_refs(abinitio_energies, self.mol_dict)
        self.abinitio_energies = abinitio_energies
        # self.rho, self.Z, self.eos_B, self.d_atoms = self.get_bulk()
        # self.Ef = get_formation_energies(abinitio_energies, ref_dict)
        # stable_adds_ids = []
        # for key in abinitio_energies:
        #    if 'slab' not in key and 'gas' not in key:
        #        stable_adds_ids.append(dbids[key])
        # self.stable_adds_ids = stable_adds_ids

    def db2atoms_info(self, fname='test_set.db',
                      selection=['series!=slab', 'layers=5', 'facet=1x1x1',
                                 'kpts=4x6']):  # selection=[]):
        c = ase.db.connect(fname)
        s = c.select(selection)
        traj = []
        for d in s:
            dbid = int(d.id)
            d = c.get(dbid)
            atoms = c.get_atoms(dbid)
            atoms.info['key_value_pairs'] = d.key_value_pairs
            traj.append(atoms)
        return traj

    def db2surf_info(self):  # selection=[]):
        c = ase.db.connect(self.slabs)
        traj = []
        for dbid in self.stable_adds_ids:
            # dbid = int(d.id)
            d = c.get(dbid)
            name = str(d.name)
            thermo_key = str(d.series) + '_' + name + '_' + str(
                d.phase) + '_' + str(d.facet)
            atoms = c.get_atoms(dbid)
            atoms.info['key_value_pairs'] = d.key_value_pairs
            atoms.info['key_value_pairs']['Ef'] = self.Ef[thermo_key]
            traj.append(atoms)
        return traj

    def db2adds_info(self, fname='example.db', selection=[]):  # selection=[]):
        c = ase.db.connect(fname)
        s = c.select(selection)
        traj = []
        for d in s:
            dbid = int(d.id)
            d = c.get(dbid)
            atoms = c.get_atoms(dbid)
            atoms.info['key_value_pairs'] = d.key_value_pairs
            traj.append(atoms)
        return traj

    def get_bulk(self):
        c = ase.db.connect(self.bulks)
        s = c.select('C=0')
        Z = {}
        rho = {}
        eos_B = {}
        d_atoms = {}
        dbcenter = {}
        dbfilling = {}
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
        return rho, Z, eos_B, dbcenter, dbfilling, d_atoms

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
            atoms.info['key_value_pairs']['Ef'] = self.Ef[str(
                d.series) + '_' + name + '_' + str(d.phase) + '_' + str(d.facet)]
            trajs.append(atoms)
        return trajs

    def elemental_dft_properties(self, atoms=None):  # , adds_dict):
        if atoms is None:
            return ['rho_vol', 'rho_A', 'B_eos']
        else:
            # series = atoms.info['key_value_pairs']['series']
            name = atoms.info['key_value_pairs']['term']
            # phase = atoms.info['key_value_pairs']['phase']
            # facet = atoms.info['key_value_pairs']['facet']
            # key = series+'_'+name+'_'+phase+'_'+facet
            A = float(atoms.cell[0, 0]) * float(atoms.cell[1, 1])
            return [
                float(self.rho[name]),
                len(atoms.numbers) / A,
                float(self.eos_B[name]),
                # float(self.dbcenter[name]),
                # float(self.dbfilling[name])
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
                    'ion_e_add1']
        else:
            metal_atoms = [a.index for a in atoms if a.symbol not in
                           ['H', 'C', 'O', 'N']]
            add_atoms = [a.index for a in atoms if a.symbol in
                         ['H', 'C', 'O', 'N']]
            liste = []
            for m in metal_atoms:
                for a in add_atoms:
                    d = atoms.get_distance(m, a, mic=True, vector=False)
                    liste.append([a, d])
            L = np.array(liste)
            i = np.argmin(L[:, 1])
            Z0 = atoms.numbers[int(L[i, 0])]
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
                 float(mnlv.ionenergies[1])
                 ]
            return result

    def primary_surfatom(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector with properties of the surface metal atom closest to an add
            atom.
        """
        if atoms is None:
            return ['Z', 'period_surf1', 'group_id_surf1',
                    # 'electron_affinity',
                    'dipole_polarizability_surf1', 'heat_of_formationsurf1',
                    # 'thermal_conductivity',
                    'specific_heat_surf1',  # 'en_allen',
                    'en_pauling_surf1', 'atomic_radius_surf1',
                    'vdw_radius_surf1', 'ion_e_surf1']
        else:
            metal_atoms = [a.index for a in atoms if a.symbol not in
                           ['H', 'C', 'O', 'N']]
            add_atoms = [a.index for a in atoms if a.symbol in
                         ['H', 'C', 'O', 'N']]
            # layer, z_layer = get_layers(atoms[metal_atoms], (0,0,1), tol=1.0)
            liste = []
            for m in metal_atoms:
                for a in add_atoms:
                    d = atoms.get_distance(m, a, mic=True, vector=False)
                    liste.append([m, d])
            L = np.array(liste)
            i = np.argmin(L[:, 1])
            Z0 = atoms.numbers[int(L[i, 0])]
            mnlv = element(Z0)
            return [Z0,
                    int(mnlv.period),
                    float(mnlv.group_id),
                    # float(mnlv.electron_affinity),
                    float(mnlv.dipole_polarizability),
                    float(mnlv.heat_of_formation),
                    # float(mnlv.thermal_conductivity),
                    float(mnlv.specific_heat),
                    # float(mnlv.en_allen),
                    float(mnlv.en_pauling),
                    float(mnlv.atomic_radius),
                    float(mnlv.vdw_radius),
                    float(mnlv.ionenergies[1]),
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
            vector containing the count of C, O, H and N atoms in the adsorbate
            first group.
        """
        if atoms is None:
            return ['nn_num_C', 'nn_num_H', 'nn_num_M']
        else:
            addsyms = ['H', 'C', 'O', 'N']
            metal_atoms = [a.index for a in atoms if a.symbol not in addsyms]
            add_atoms = [a.index for a in atoms if a.symbol in addsyms]
            liste = []
            for m in metal_atoms:
                for a in add_atoms:
                    d = atoms.get_distance(m, a, mic=True, vector=False)
                    liste.append([a, d])
            L = np.array(liste)
            i = np.argmin(L[:, 1])
            primary_add = int(L[i, 0])
            nH1 = len([a.index for a in atoms if a.symbol == 'H' and
                       atoms.get_distance(primary_add, a.index, mic=True) <
                       1.3 and a.index != primary_add])
            nC1 = len([a.index for a in atoms if a.symbol == 'C' and
                       atoms.get_distance(primary_add, a.index, mic=True) <
                       1.3 and a.index != primary_add])
            nM = len([a.index for a in atoms if a.symbol not in addsyms and
                      atoms.get_distance(primary_add, a.index, mic=True) <
                      2.35])
            return [nC1, nH1, nM]  # , nN, nH]

    def secondary_adds_nn(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector containing the count of C, O, H and N atoms in the adsorbate
            first group.

            This function is relevant for adsorbates that bind through 2 atoms.
            Example: CH2CH2 on Pt111 binds through 2 C atoms
        """
        if atoms is None:
            return ['num_C_add2nn', 'num_H_add2nn', 'num_M_add2nn']
        else:
            metal_atoms = [a.index for a in atoms if a.symbol not in
                           ['H', 'C', 'O', 'N']]
            add_atoms = [a.index for a in atoms if a.symbol in
                         ['H', 'C', 'O', 'N']]
            liste = []
            for m in metal_atoms:
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
            nM = len([a.index for a in atoms if a.symbol in metal_atoms and
                     atoms.get_distance(secondary_add, a.index, mic=True) <
                     2.35])
            return [nC1, nH1, nM]  # , nN, nH]

    def primary_surf_nn(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector containing the count of nearest neighbors and properties of
            the nearest neighbors.
        """
        if atoms is None:
            return ['num_nn', 'nn_num_identical', 'av_dbcenter',
                    'av_dbfilling', 'av_heat_of_formation']
        else:
            metal_atoms = [a.index for a in atoms if a.symbol not in
                           ['H', 'C', 'O', 'N']]
            add_atoms = [a.index for a in atoms if a.symbol in
                         ['H', 'C', 'O', 'N']]
            liste = []
            for m in metal_atoms:
                for a in add_atoms:
                    d = atoms.get_distance(m, a, mic=True, vector=False)
                    liste.append([m, d])
            L = np.array(liste)
            i = np.argmin(L[:, 1])
            primary_surf = int(L[i, 0])
            symbols = atoms.get_chemical_symbols()
            name = symbols[primary_surf]
            r_bond = float(element(name).atomic_radius) * 2.3E-2
            ai = np.where(atoms.get_distances(primary_surf, metal_atoms,
                                              mic=True) < r_bond)
            if name in self.dbcenter:
                dbcenter = [self.dbcenter[name]]
                dbfilling = [self.dbfilling[name]]
            else:
                dbcenter = []
                dbfilling = []
            heat_of_formation = [element(name).heat_of_formation]
            q_self = []
            for q in ai[0]:
                sym = symbols[metal_atoms[q]]
                mnlv = element(sym)
                if sym in self.dbcenter:
                    dbcenter.append(self.dbcenter[sym])
                    dbfilling.append(self.dbfilling[sym])
                    heat_of_formation.append(mnlv.heat_of_formation)
                if sym == name:
                    q_self.append(q)
            av_dbcenter = np.average(dbcenter)
            av_dbfilling = np.average(dbfilling)
            av_heat_of_formation = np.average(heat_of_formation)
            n = len(ai)
            n_self = len(q_self)
            return [n, n_self, av_dbcenter, av_dbfilling, av_heat_of_formation]

    def adds_sum(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector containing sums of the atomic properties of the adsorbate.
        """
        if atoms is None:
            return ['sum_electron_affinity', 'av_electron_affinity',
                    'sum_en_allen', 'av_en_allen', 'sum_en_pauling',
                    'av_en_pauling']
        else:
            add_atoms = [a.index for a in atoms if a.symbol in
                         ['H', 'C', 'O', 'N']]
            electron_affinity = 0
            en_allen = 0
            en_pauling = 0
            heat_of_formation = 0
            L = len(atoms)
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
        if atoms is None:
            return ['random']
        else:
            return [random()]

    def get_Ef(self, atoms=None):
        if atoms is None:
            return ['Ef']
        else:
            Ef = 0
            name = str(atoms.info['key_value_pairs']['name'])
            series = str(atoms.info['key_value_pairs']['series'])
            composition = string2symbols(series)
            for a in composition:
                Ef -= self.ref_dict[a]
            # Ef = float(atoms.info['key_value_pairs']['Ef'])
            thermo_key = 'slab_' + name + '_' + str(atoms.info['key_value_pairs']['phase']) + '_' + str(atoms.info['key_value_pairs']['surf_lattice'])
            try:
                Ef += float(atoms.info['key_value_pairs']['enrgy']) - self.abinitio_energies[thermo_key]
            except KeyError:
                Ef = np.NaN
                warnings.warn(thermo_key+' not found. get_Ef returns NaN')
            return [Ef]

    def get_keyvaluepair(self, atoms=None, field_name='None'):
        if atoms is None:
            return ['kvp_'+field_name]
        else:
            field_value = float(atoms['key_value_pairs'][field_name])
            [field_value]
            return
