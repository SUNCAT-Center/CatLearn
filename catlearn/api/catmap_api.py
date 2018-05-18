"""Functions that interface CatLearn with CatMAP."""
import ase.db
from catmap.ase_data import db2catmap


def catmap_energy(fname, database_ids, prediction,
                  uncertainty=None, catmap=None):
    """Return CatMAP ase_data object with predicted formation energies.

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

    if catmap is None:
        catmap = db2catmap()
        catmap.formation_energies = {}
        catmap.dbid = {}
        catmap.std = {}

    formation_energies = {}
    dbids = {}
    std = {}
    for i, dbid in enumerate(database_ids):
        d = c.get(dbid)
        [n, name, phase, surf_lattice, facet,
         cell] = catmap._get_adsorbate_fields(d)
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
    catmap.formation_energies.update(formation_energies)
    catmap.std.update(std)
    catmap.dbid.update(dbids)

    return catmap
