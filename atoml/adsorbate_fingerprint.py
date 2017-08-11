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
from .general_fingerprint import get_mendeleev_params
from .db2thermo import layers_info


def n_outer(econf):
    n_tot = 0
    ns = 0
    np = 0
    nd = 0
    nf = 0
    for shell in econf.split(' ')[1:]:
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
                try:
                    bulkcomp = string2symbols(name)
                    ldbcenter = []
                    ldbfilling = []
                    ldbwidth = []
                    ldbskew = []
                    ldbkurt = []
                    for nam in bulkcomp:
                        try:
                            ldbkurt.append(float(self.dbkurt[nam]))
                            ldbfilling.append(float(self.dbfilling[nam]))
                            ldbwidth.append(float(self.dbwidth[nam]))
                            ldbskew.append(float(self.dbskew[nam]))
                            ldbkurt.append(float(self.dbkurt[nam]))
                        except KeyError:
                            continue
                    dbcenter = np.average(ldbcenter)
                    dbfilling = np.average(ldbfilling)
                    dbwidth = np.average(ldbwidth)
                    dbskew = np.average(ldbskew)
                    dbkurt = np.average(ldbkurt)
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
                    'ne_outer_add1',
                    'ne_s_add1',
                    'ne_p_add1',
                    'ne_d_add1',
                    'ne_f_add1',
                    'ground_state_magmom_add1']
        else:
            # Get atomic number of alpha adsorbate atom.
            Z0 = int(atoms.info['Z_add1'])
            # Import AtoML data on that element.
            dat = get_mendeleev_params(Z0, extra_params=['econf'])
            result = dat[:-1] + list(n_outer(dat[-1]))
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
                    'ne_outer_surf1',
                    'ne_s_surf1',
                    'ne_p_surf1',
                    'ne_d_surf1',
                    'ne_f_surf1',
                    'dbcenter_surf1',
                    'dbfilling_surf1',
                    'dbwidth_surf1',
                    'dbskew_surf1',
                    'dbkurtosis_surf1',
                    'ground_state_magmom_surf1']
        else:
            Z0 = int(atoms.info['Z_surf1'])
            dat = get_mendeleev_params(Z0, extra_params=['econf'])
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
            result = dat[:-1] + list(n_outer(dat[-1]))
            return result + [dbcenter,
                             dbfilling,
                             dbwidth,
                             dbskew,
                             dbkurt,
                             float(ground_state_magnetic_moments[Z0])]

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
            dH = covalent_radii[1]
            dC = covalent_radii[6]
            dM = covalent_radii[Z_surf1]
            dadd = covalent_radii[Z_add1]
            nH1 = len([a.index for a in atoms if a.symbol == 'H' and
                       atoms.get_distance(primary_add, a.index, mic=True) <
                       (dH+dadd)*1.15 and a.index != primary_add])
            nC1 = len([a.index for a in atoms if a.symbol == 'C' and
                       atoms.get_distance(primary_add, a.index, mic=True) <
                       (dC+dadd)*1.15 and a.index != primary_add])
            nM = len([a.index for a in atoms if a.symbol not in addsyms and
                      atoms.get_distance(primary_add, a.index, mic=True) <
                      (dM+dadd)*1.15])
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
                       1.15 and a.index != secondary_add])
            nC1 = len([a.index for a in atoms if a.symbol == 'C' and
                      atoms.get_distance(secondary_add, a.index, mic=True) <
                      1.15 and a.index != secondary_add])
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
            return ['nn_surf1', 'identnn_surf1',
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
                    'ne_outer_surf2',
                    'ne_s_surf2',
                    'ne_p_surf2',
                    'ne_d_surf2',
                    'ne_f_surf2',
                    'ground_state_magmom_surf2',
                    'dbcenter_surf2',
                    'dbfilling_surf2',
                    'dbwidth_surf2',
                    'dbskew_surf2',
                    'dbkurtosis_surf2']
        else:
            add_atoms = atoms.info['add_atoms']
            if atoms.info['key_value_pairs']['supercell'] == '1x1':
                atoms = atoms.repeat([2, 2, 1])
            elif atoms.info['key_value_pairs']['supercell'] == '3x2':
                atoms = atoms.repeat([1, 2, 1])
            bulk_atoms, top_atoms = layers_info(atoms)
            liste = []
            for m in top_atoms:
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
            for nni in top_atoms:
                d_nn = atoms.get_distance(primary_surf, nni, mic=True,
                                          vector=False)
                ai.append([nni, d_nn])
            # ai = np.where(atoms.get_distances(primary_surf, surf_atoms,
            #                                  mic=True) < r_bond)
            try:
                dbcenter = float(self.dbcenter[name])
                dbfilling = float(self.dbfilling[name])
                dbwidth = float(self.dbwidth[name])
                dbskew = float(self.dbskew[name])
                dbkurt = float(self.dbkurt[name])
                db = [[dbcenter, dbfilling, dbwidth, dbskew, dbkurt]]
            except KeyError:
                db = [[np.NaN]*5]
            mnlv0 = get_mendeleev_params(Z0, extra_params=['econf'])
            dat = [mnlv0[:-1] + list(n_outer(mnlv0[-1])) +
                   [float(ground_state_magnetic_moments[Z0])]]
            n = 0
            q_self = []
            for nn in ai:
                q = nn[0]
                Znn = numbers[q]
                r_bond_nn = covalent_radii[Znn]
                if q != primary_surf and 1.15*(r_bond_nn+r_bond) > nn[1]:
                    sym = symbols[q]
                    mnlv = get_mendeleev_params(q, extra_params=['econf'])
                    n += 1
                    if sym in self.dbcenter:
                        dbcenter = float(self.dbcenter[sym])
                        dbfilling = float(self.dbfilling[sym])
                        dbwidth = float(self.dbwidth[sym])
                        dbskew = float(self.dbskew[sym])
                        dbkurt = float(self.dbkurt[sym])
                        db.append([dbcenter, dbfilling, dbwidth, dbskew,
                                   dbkurt])
                    else:
                        db.append(5*[np.NaN])
                    dat.append(mnlv[:-1] + list(n_outer(mnlv[-1])) +
                               [float(ground_state_magnetic_moments[q])])
                    if sym == name:
                        q_self.append(q)
            n_self = len(q_self)
            return [n, n_self] + list(np.nanmean(dat, axis=0, dtype=float)) + \
                list(np.nanmean(db, axis=0, dtype=float))

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
                    'ne_f_add_av']
        else:
            add_atoms = atoms.info['add_atoms']
            dat = []
            for a in add_atoms:
                Z = atoms.numbers[a]
                mnlv = get_mendeleev_params(Z, extra_params=['econf'])
                dat.append(mnlv[:-1] + list(n_outer(mnlv[-1])))
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
