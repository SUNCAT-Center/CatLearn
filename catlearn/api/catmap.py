"""API for CatMAP."""
import ase.db
import json


def catmap_energy_landscape(fname, database_ids, prediction,
                            uncertainty=None, catmap=None,
                            site_specific=False):
    """Return CatMAP ase_data object with predicted formation energies.

    Parameters
    ----------
        fname : str
            Path and filename of candidate ase database file.
        database_ids : list
            Database ids.
        prediction : list
            Predicted means in the same order as database_ids.
        uncertainty : list
            Predicted uncertainties in the same order as database_ids.
        catmap : object
            CatMAP energy_landscape object.
        site_specific : bool
            If True: Dinstinguish sites using the site key value pair, and
            stores a the potential energy of adsorbates on each site.
            Else: Use the minimum ab initio energy, disregarding the site.
    """
    c = ase.db.connect(fname)

    # if catmap is None:
    #    catmap = EnergyLandscape()
    #    catmap.formation_energies = {}
    #    catmap.dbid = {}
    #    catmap.std = {}

    formation_energies = {}
    dbids = {}
    std = {}
    for i, dbid in enumerate(database_ids):
        d = c.get(dbid)
        n, species, name, phase, surf_lattice, facet, cell = \
            catmap._get_adsorbate_fields(d)
        if site_specific and 'site' in d:
            site = str(d.site)
        else:
            site = 'site'
        key = '_'.join([str(n), species, name, phase, surf_lattice,
                        facet, cell, site])
        if key not in formation_energies:
            formation_energies[key] = prediction[i]
            std[key] = uncertainty[i]
            dbids[key] = dbid
        elif formation_energies[key] < prediction[i]:
            formation_energies[key] = prediction[i]
            std[key] = uncertainty[i]
            dbids[key] = dbid
    catmap.formation_energies.update(formation_energies)
    catmap.std.update(std)
    catmap.dbid.update(dbids)

    return catmap


def catmap_pickle(fname):
    f = open(fname, 'r')
    catmap_model = json.load(f)
    return catmap_model


def get_rate_control(state, catmap_model):
    n, species, cat_name, lattice, cell, site = state.split('_')
    # catmap_model
    return species
