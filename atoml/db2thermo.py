# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:04:38 2017

This function constructs a dictionary with abinitio_energies.

Input: 
    fname (str) path/filename of ase.db file
    selection (list) ase.db selection

@author: mhangaard
"""

import ase.db


def db2surf(fname, selection=[]):
    csurf = ase.db.connect(fname)
    ssurf = csurf.select(selection)
    abinitio_energies = {}
    dbids = {}
    for d in ssurf:                 #get slabs and adsorbates from .db
        cat = str(d.name)+'_'+str(d.phase)
        site_name = str(d.surf_lattice)
        abinitio_energy = float(d.enrgy)
        #composition=str(d.formula)
        series = str(d.series)
        if series+'_'+cat+'_'+site_name not in abinitio_energies:
            abinitio_energies[series+'_'+cat+'_'+site_name] = abinitio_energy
            dbids[series+'_'+cat+'_'+site_name] = int(d.id)
        elif abinitio_energies[series+'_'+cat+'_'+site_name] > abinitio_energy:
            abinitio_energies[series+'_'+cat+'_'+site_name] = abinitio_energy
            dbids[series+'_'+cat+'_'+site_name] = int(d.id)
    return abinitio_energies, dbids
    
def db2mol(fname,selection=[]): #fname must be path/filename of db containing molecules
    cmol = ase.db.connect(fname)
    smol = cmol.select(selection)
    #mol_dict = {}
    abinitio_energies = {}
    dbids = {}
    for d in smol:              #get molecules from mol.db
        abinitio_energy = float(d.enrgy)
        species_name=str(d.formula)
        if species_name+'_gas' not in abinitio_energies:
            abinitio_energies[species_name+'_gas'] = abinitio_energy
            dbids[species_name+'_gas'] = int(d.id)
        elif abinitio_energies[species_name+'_gas'] > abinitio_energy:
            abinitio_energies[species_name+'_gas'] = abinitio_energy
            dbids[species_name+'_gas'] = int(d.id)
    return abinitio_energies, dbids

def mol2ref(abinitio_energies):
    mol_dict = {}
    mol_dict['H'] = 0.5*abinitio_energies['H2_gas']
    mol_dict['O'] = abinitio_energies['H2O_gas'] - 2*mol_dict['H']
    mol_dict['C'] = abinitio_energies['CH4_gas'] - 4*mol_dict['H']
    #mol_dict['C'] = abinitio_energies['CO_gas'] - mol_dict['O']
    return mol_dict
    
def get_refs(energy_dict,energy_mols): #adapted from CATMAP wiki
    ref_dict = energy_mols
    for key in energy_dict.keys():
        if 'slab' in key:
            ser,cat,pha,fac = key.split('_')
            Eref = energy_dict[key]
            name = cat+'_'+pha+'_'+fac
            ref_dict[name] = Eref
    return ref_dict