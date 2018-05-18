"""Functions that interface CatLearn with CatMAP."""
import ase.db
from catmap.ase_data import db2catmap


def catmap_dict(database_ids, prediction, uncertainty, fname):
    """Return CatMAP ase_api object with predicted formation energies.

    Parameters
    ----------
        database_ids : list
            Database ids.
        prediction : list
            Predicted means.
        uncertainty : list
            Predicted uncertainties.
        fname : str
            Path and filename of ase database file.
    """
    c = ase.db.connect(fname)
    catmap_api = db2catmap()
    formation_energies = {}
    dbids = {}
    std = {}
    for i, dbid in enumerate(database_ids):
        d = c.get(dbid)
        [n, name, phase, surf_lattice, facet,
         cell] = catmap_api._get_adsorbate_fields(d)
        key = '_'.join([n, str(d.species), name, phase, surf_lattice,
                        facet, cell, str(d.site)])
        if key not in formation_energies:
            formation_energies[key] = prediction[i]
            std[key] = uncertainty[i]
            dbids[key] = dbid[i]
        elif formation_energies[key] < prediction[i]:
            formation_energies[key] = prediction[i]
            std[key] = uncertainty[i]
            dbids[key] = dbid[i]
    catmap_api.formation_energies = formation_energies
    catmap_api.std = std
    catmap_api.dbid = dbids
    return catmap_api
