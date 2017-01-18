# -*- coding: utf-8 -*-
"""
Slab adsorbate fingerprint functions for machine learning

Created on Tue Dec  6 14:09:29 2016

@author: mhangaard

"""

import numpy as np
from ase import db
from db2thermo import db2mol, db2surf, mol2ref, get_refs, get_formation_energies
from mendeleev import element
from random import random

class AdsorbateFingerprintGenerator(object):
    def __init__(self, mols='mol.db', bulks='ref_bulks.db', slabs=None):
        self.mols = mols
        self.bulks = bulks
        self.rho, self.Z, self.eos_B, self.dbcenter, self.dbfilling, self.d_atoms = self.get_bulk() 
        self.slabs = slabs
        if slabs != None:
            abinitio_energies, frequency_dict, contribs, dbids = db2mol(mols, ['vacuum=8','PW=500'])
            surf_energies, surf_frequencies, surf_contribs, surf_dbids = db2surf(slabs, ['Al=0','C=0','layers=5','facet=1x1x1','kpts=4x6','PW=500','PSP=gbrv1.5pbe'])
            abinitio_energies.update(surf_energies)
            frequency_dict.update(surf_frequencies)
            contribs.update(surf_contribs)
            dbids.update(surf_dbids)
            #surf_energies211, surf_frequencies211, surf_contribs211, surf_dbids211 = db2surf(slabs, ['Al=0','Rh=0','Pt=0','layers=4','facet=2x1x1','kpts=4x4','PW=500','PSP=gbrv1.5pbe'])
            #abinitio_energies.update(surf_energies211)
            #frequency_dict.update(surf_frequencies211)
            #contribs.update(surf_contribs211)
            #dbids.update(surf_dbids211)
            self.mol_dict = mol2ref(abinitio_energies)
            ref_dict = get_refs(abinitio_energies, self.mol_dict)
            #self.rho, self.Z, self.eos_B, self.d_atoms = self.get_bulk()
            self.Ef = get_formation_energies(abinitio_energies, ref_dict)
            stable_adds_ids = []
            for key in abinitio_energies:
                if 'slab' not in key and 'gas' not in key:
                    stable_adds_ids.append(dbids[key])
            self.stable_adds_ids = stable_adds_ids
    
    def db2atoms_info(self, fname='test_set.db', selection=['series!=slab','C=0','layers=5','facet=1x1x1','kpts=4x6']): #selection=[]):
        c = db.connect(fname)
        s = c.select(selection)
        traj = []
        for d in s:    
            dbid = int(d.id)
            d = c.get(dbid)
            atoms = c.get_atoms(dbid)
            atoms.info['key_value_pairs'] = d.key_value_pairs
            traj.append(atoms)
        return traj
    
    def db2surf_info(self): #selection=[]):
        c = db.connect(self.slabs)
        traj = []
        for dbid in self.stable_adds_ids:    
            #dbid = int(d.id)
            d = c.get(dbid)            
            name = str(d.name)
            thermo_key = str(d.series)+'_'+name+'_'+str(d.phase)+'_'+str(d.facet)
            atoms = c.get_atoms(dbid)
            atoms.info['key_value_pairs'] = d.key_value_pairs
            atoms.info['key_value_pairs']['Ef'] = self.Ef[thermo_key]
            traj.append(atoms)
        return traj
    
    def get_bulk(self):
        c = db.connect(self.bulks)
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
            d_atoms[name] = (atoms.get_distances(0, range(1,len(atoms)), mic=True)).min()
            if 'dbcenter' in d.key_value_pairs:
                dbcenter[name] = float(d.dbcenter)
                dbfilling[name] = float(d.dbfilling)
        return rho, Z, eos_B, dbcenter, dbfilling, d_atoms
        
    def db2atoms_fps(self):#, adds_dict):
        c = db.connect(self.slabs)
        s = c.select(['series!=slab','C=0','layers=5','facet=1x1x1','kpts=4x6','PW=500','PSP=gbrv1.5pbe'])
        trajs = []
        for d in s:
            dbid = int(d.id)
            name = str(d.name)
            atoms = c.get_atoms(dbid)
            atoms.info['key_value_pairs'] = {}#d.key_value_pairs
            atoms.info['key_value_pairs']['Z'] = self.Z[name]
            atoms.info['key_value_pairs']['rho_bulk'] = self.rho[name]
            A = float(d.cell[0,0]) * float(d.cell[1,1])
            atoms.info['key_value_pairs']['rho_surf'] = len(d.numbers) / A
            atoms.info['key_value_pairs']['Ef'] = self.Ef[str(d.series)+'_'+name+'_'+str(d.phase)+'_'+str(d.facet)]
            trajs.append(atoms)
        return trajs
        
    def elemental_dft_properties(self, atoms=None):#, adds_dict):
        if atoms==None:
            return ['rho_vol', 'rho_A', 'B_eos']
        else:
            #series = atoms.info['key_value_pairs']['series']
            name = atoms.info['key_value_pairs']['term']
            #phase = atoms.info['key_value_pairs']['phase']
            #facet = atoms.info['key_value_pairs']['facet']
            #key = series+'_'+name+'_'+phase+'_'+facet
            A = float(atoms.cell[0,0]) * float(atoms.cell[1,1])
            return [
                float(self.rho[name]), 
                len(atoms.numbers) / A, 
                float(self.eos_B[name]),
                #float(self.dbcenter[name]),
                #float(self.dbfilling[name])
                ]
        
    def primary_addatom(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector containing properties of the closest add atom to a surface metal atom.
        """
        if atoms==None:
            return ['Z','period','group_id','electron_affinity',
                'dipole_polarizability',#'en_allen', 
                'en_pauling',
                'atomic_radius','vdw_radius','ion_e']
        else:
            metal_atoms = [a.index for a in atoms if a.symbol not in ['H','C','O','N']]
            add_atoms = [a.index for a in atoms if a.symbol in ['H','C','O','N']]
            liste = []
            for m in metal_atoms:
                for a in add_atoms:
                    d = atoms.get_distance(m, a, mic=True, vector=False)
                    liste.append([a,d])
            L = np.array(liste)
            i = np.argmin(L[:,1])
            Z0 = atoms.numbers[int(L[i,0])]
            mendeleev = element(Z0)
            result = [
                 Z0,
                 int(mendeleev.period), 
                 float(mendeleev.group_id), 
                 float(mendeleev.electron_affinity), 
                 float(mendeleev.dipole_polarizability),
                 #float(mendeleev.en_allen),
                 float(mendeleev.en_pauling),
                 float(mendeleev.atomic_radius),
                 float(mendeleev.vdw_radius),
                 float(mendeleev.ionenergies[1])
                 ]
            return result
    
    def primary_surfatom(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector with properties of the surface metal atom closest to an add atom.
        """
        if atoms==None:
            return ['Z','period_surf1','group_id_surf1',
                 #'electron_affinity',
                 'dipole_polarizability_surf1','heat_of_formationsurf1',
                 #'thermal_conductivity',
                 'specific_heat_surf1',#'en_allen',
                 'en_pauling_surf1','atomic_radius_surf1','vdw_radius_surf1','ion_e_surf1']
        else:
            metal_atoms = [a.index for a in atoms if a.symbol not in ['H','C','O','N']]
            add_atoms = [a.index for a in atoms if a.symbol in ['H','C','O','N']]
            #layer, z_layer = get_layers(atoms[metal_atoms], (0,0,1), tol=1.0)
            liste = []
            for m in metal_atoms:
                for a in add_atoms:
                    d = atoms.get_distance(m, a, mic=True, vector=False)
                    liste.append([m,d])
            L = np.array(liste)
            i = np.argmin(L[:,1])
            Z0 = atoms.numbers[int(L[i,0])]
            mendeleev = element(Z0)
            return [Z0, 
                    int(mendeleev.period), 
                    float(mendeleev.group_id), 
                    #float(mendeleev.electron_affinity), 
                    float(mendeleev.dipole_polarizability),
                    float(mendeleev.heat_of_formation),
                    #float(mendeleev.thermal_conductivity),
                    float(mendeleev.specific_heat),
                    #float(mendeleev.en_allen),
                    float(mendeleev.en_pauling),
                    float(mendeleev.atomic_radius),
                    float(mendeleev.vdw_radius),
                    float(mendeleev.ionenergies[1]),
                    ]
    
    def Z_add(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector containing the count of C, O, H and N atoms in the adsorbate.
        """
        if atoms==None:
            return ['total_num_C', 'total_num_H']
        else:
            nC = len([a.index for a in atoms if a.symbol == 'C'])
            #nO = len([a.index for a in atoms if a.symbol == 'O'])
            #nN = len([a.index for a in atoms if a.symbol == 'N'])
            nH = len([a.index for a in atoms if a.symbol == 'H'])
            return [nC, nH]#, nN, nO]
        
    def primary_adds_nn(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector containing the count of C, O, H and N atoms in the adsorbate first group.
        """
        if atoms==None:
            return ['nn_num_C', 'nn_num_H', 'nn_num_M']
        else:
            metal_atoms = [a.index for a in atoms if a.symbol not in ['H','C','O','N']]
            add_atoms = [a.index for a in atoms if a.symbol in ['H','C','O','N']]
            liste = []
            for m in metal_atoms:
                for a in add_atoms:
                    d = atoms.get_distance(m, a, mic=True, vector=False)
                    liste.append([a,d])
            L = np.array(liste)
            i = np.argmin(L[:,1])
            primary_add = int(L[i,0])
            nH1 = len([a.index for a in atoms if a.symbol == 'H' and 
                atoms.get_distance(primary_add,a.index, mic=True) < 1.3 and
                a.index != primary_add])
            nC1 = len([a.index for a in atoms if a.symbol == 'C' and 
                atoms.get_distance(primary_add,a.index, mic=True) < 1.3 and
                a.index != primary_add])
            nM = len([a.index for a in atoms if a.symbol in metal_atoms and 
                atoms.get_distance(primary_add,a.index, mic=True) < 2.35])
            return [nC1, nH1, nM]#, nN, nH]
    
    def secondary_adds_nn(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector containing the count of C, O, H and N atoms in the adsorbate first group.
            This function is relevant for adsorbates that bind through 2 atoms.
            Example: CH2CH2 on Pt111 binds through 2 C atoms 
        """
        if atoms==None:
            return ['num_C_add2nn', 'num_H_add2nn', 'num_M_add2nn']
        else:
            metal_atoms = [a.index for a in atoms if a.symbol not in ['H','C','O','N']]
            add_atoms = [a.index for a in atoms if a.symbol in ['H','C','O','N']]
            liste = []
            for m in metal_atoms:
                for a in add_atoms:
                    d = atoms.get_distance(m, a, mic=True, vector=False)
                    liste.append([a,d])
            L = np.array(liste)
            i = np.argsort(L[:,1])[1]
            secondary_add = int(L[i,0])
            nH1 = len([a.index for a in atoms if a.symbol == 'H' and 
                atoms.get_distance(secondary_add,a.index, mic=True) < 1.3 and
                a.index != secondary_add])
            nC1 = len([a.index for a in atoms if a.symbol == 'C' and 
                atoms.get_distance(secondary_add,a.index, mic=True) < 1.3 and
                a.index != secondary_add])
            nM = len([a.index for a in atoms if a.symbol in metal_atoms and 
                atoms.get_distance(secondary_add,a.index, mic=True) < 2.35])
            return [nC1, nH1, nM]#, nN, nH]    
    
    def primary_surf_nn(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector containing the count of nearest neighbors.
        """
        if atoms==None:
            return ['num_nn','av_dbcenter', 'av_dbfilling']
        else:
            metal_atoms = [a.index for a in atoms if a.symbol not in ['H','C','O','N']]
            add_atoms = [a.index for a in atoms if a.symbol in ['H','C','O','N']]
            liste = []
            for m in metal_atoms:
                for a in add_atoms:
                    d = atoms.get_distance(m, a, mic=True, vector=False)
                    liste.append([m,d])
            L = np.array(liste)
            i = np.argmin(L[:,1])
            primary_surf = int(L[i,0])
            symbols = atoms.get_chemical_symbols()
            name = symbols[primary_surf]
            r_bond = self.d_atoms[name]*1.15
            ai = np.where( atoms.get_distances(primary_surf, metal_atoms, 
                                                  mic=True) < r_bond)
            dbcenter = [self.dbcenter[name]]
            dbfilling = [self.dbfilling[name]]
            for q in ai[0]:
                sym = symbols[metal_atoms[q]]
                if sym in self.dbcenter:
                    dbcenter.append(self.dbcenter[sym])
                    dbfilling.append(self.dbfilling[sym])
            av_dbcenter = np.average(dbcenter)
            av_dbfilling = np.average(dbfilling)
            n = len(ai)
            return [n, av_dbcenter, av_dbfilling]
    
    def adds_sum(self, atoms=None):
        """ Function that takes an atoms objects and returns a fingerprint
            vector containing sums of the atomic properties of the adsorbate.
        """
        if atoms==None:
            return ['sum_electron_affinity', 'average_electron_affinity',
                 'sum_en_allen', 'average_en_allen', 
                 'sum_en_pauling', 'average_en_pauling']
        else:
            add_atoms = [a.index for a in atoms if a.symbol in ['H','C','O','N']]
            electron_affinity = 0
            en_allen = 0
            en_pauling = 0
            for a in add_atoms:
                Z0 = atoms.numbers[a]
                electron_affinity += float(element(Z0).electron_affinity)
                en_allen += float(element(Z0).en_allen)
                en_pauling += float(element(Z0).en_pauling)
            result = [electron_affinity, electron_affinity/len(add_atoms), 
                en_allen, en_allen/len(add_atoms), en_pauling, 
                en_pauling/len(add_atoms)]
            return result
    
    def randomfpv(self, atoms=None):
        if atoms==None:
            return ['random']
        else:
            return [random()]
    
    def get_Ef(self, atoms=None):
        if atoms==None:
            return ['Ef']
        else:
            Ef = float(atoms.info['key_value_pairs']['Ef'])
            return [Ef]

        








