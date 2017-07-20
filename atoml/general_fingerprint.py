from ase.db import connect


def get_mendeleev_params(d, index=None, extra_params=[]):
    """ Return a list of generic parameters about
    an indexed atom from an ase-database entry.

    Parameters
    ----------
    d : ase-database instance, str, int
        An ase-database entry, atomic symbol, or atomic number.
    index : int
        The atom index whose parameters to return.
    extra_params: list of str
	Extra Mendaleev parameters to be returned in
        the list. For a full list see here:
	http://nbviewer.jupyter.org/url/bitbucket.org/lukaszmentel/mendeleev/raw/tip/docs/ipynb/tables.ipynb

    Returns
    -------
    list
        All parameters of the indexed element in the
        params list.

    """

    # Parameters which typically do not contain None
    params = [
	'atomic_number',
	'atomic_volume',
	'boiling_point',
	'density',
	'dipole_polarizability',
	'electron_affinity',
	'group_id',
	'lattice_constant',
	'melting_point',
	'period',
	'vdw_radius',
	'covalent_radius_cordero',
	'en_allen',
	'atomic_weight']

    # Pull parameters from a proxy database for
    # improved performance.
    db = connect('../data/proxy-mendeleev.db')

    if extra_params:
	params += extra_params

    if isinstance(d, str) or isinstance(d, int):
        sym = d
    else:
        sym = d.symbols[index]

    var = []
    for _ in params:
	var += [db.get(sym).data.get(_)]

    return var
