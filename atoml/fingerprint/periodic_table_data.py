"""Function pulling atomic data for elements."""
import json
from atoml import __path__ as atoml_path

# Load the Mendeleev parameter data into memory
with open('/'.join(atoml_path[0].split('/')[:-1]) +
          '/atoml/data/proxy-mendeleev.json') as f:
    data = json.load(f)


def get_mendeleev_params(atomic_number, extra_params=[]):
    """Return a list of generic parameters about an atom.

    Parameters
    ----------
    atomic_number : list or int
        An atomic number.
    extra_params: list of str
        Extra Mendaleev parameters to be returned in the list.
        For a full list see here - https://goo.gl/G4eTvu

    Returns
    -------
    var : list
        All parameters of the element with specified atomic number.
    """
    # Type check atomic_number var. Switch to list if needed.
    if type(atomic_number) is int:
        atomic_number = [atomic_number]

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
    for an in atomic_number:
        for _ in params:
            var += [data[str(an)].get(_)]

    return var
