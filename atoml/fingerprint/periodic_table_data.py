import json
from atoml import __path__ as atoml_path

# Load the Mendeleev parameter data into memory
with open('/'.join(atoml_path[0].split('/')[:-1]) +
          '/atoml/data/proxy-mendeleev.json') as f:
    data = json.load(f)


def get_mendeleev_params(atomic_number, index=None, extra_params=[]):
    """ Return a list of generic parameters about
    an indexed atom from an ase-database entry.

    Parameters
    ----------
    atomic_number : int
        An atomic number.
    index : int
        The atom index whose parameters to return.
    extra_params: list of str
        Extra Mendaleev parameters to be returned in the list.
        For a full list see here:
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

    if extra_params:
        params += extra_params

    var = []
    for _ in params:
        var += [data[str(atomic_number)].get(_)]

    return var
