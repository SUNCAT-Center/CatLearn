"""Some wrapper functions for asap3."""
from asap3.analysis import PTM


def ptm_structure_fpv(self, atoms):
    """Polyhedral Template Matching wrapper for ASAP.

    Parameters
    ----------
    atoms : object
        An ase atoms object.
    """
    msg = "ASAP must be installed to use this function:"
    msg += " https://wiki.fysik.dtu.dk/asap"
    ptmdata = PTM(atoms)

    return ptmdata['structure']


def ptm_alloy_fpv(self, atoms):
    """Polyhedral Template Matching wrapper for ASAP.

    Parameters
    ----------
    atoms : object
        An ase atoms object.
    """
    msg = "ASAP must be installed to use this function:"
    msg += " https://wiki.fysik.dtu.dk/asap"
    ptmdata = PTM(atoms)

    return ptmdata['alloytype']
