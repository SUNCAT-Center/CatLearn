"""Function pulling atomic data for elements.

This is typically used in conjunction with other fiungerprint generators to
combine general atomic data with more specific properties.
"""
import json
from catlearn import __path__ as catlearn_path
from ase.data import covalent_radii
import numpy as np

# Load the Mendeleev parameter data into memory
with open('/'.join(catlearn_path[0].split('/')[:-1]) +
          '/catlearn/data/proxy-mendeleev.json') as f:
    data = json.load(f)

block2number = {'s': 1,
                'p': 2,
                'd': 3,
                'f': 4}

default_params = ['atomic_number',
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

electronegativities = ['en_allen',
                       'en_nagle',
                       'en_gordy',
                       'en_martynov-batsanov',
                       'en_allred-rochow',
                       'en_cottrell-sutton',
                       'en_pauling',
                       'en_mulliken']


def n_outer(econf):
    n_tot = 0
    n_s = 0
    n_p = 0
    n_d = 0
    n_f = 0
    for shell in econf.split(' ')[1:]:
        n_shell = 0
        if shell[-1].isalpha():
            n_shell = 1
        elif len(shell) == 3:
            n_shell = int(shell[-1])
        elif len(shell) == 4:
            n_shell = int(shell[-2:])
        n_tot += n_shell
        if 's' in shell:
            n_s += n_shell
        elif 'p' in shell:
            n_p += n_shell
        elif 'd' in shell:
            n_d += n_shell
        elif 'f' in shell:
            n_f += n_shell
    return n_tot, n_s, n_p, n_d, n_f


def get_mendeleev_params(atomic_number, params=None):
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
    try:
        atomic_number = int(atomic_number)
    except TypeError:
        pass
    if type(atomic_number) is int:
        atomic_number = [atomic_number]

    # Parameters which typically do not contain None
    if params is None:
        params = default_params

    var = []
    for an in atomic_number:
        for _ in params:
            var += [data[str(an)].get(_)]

    return var


def list_mendeleev_params(numbers, params=None):
    """Return an n by p array, containing p parameters of n atoms.

    Parameters
    ----------
        numbers : list
            atomic numbers.
        params : list
            elemental parameters.
    """
    if params is None:
        params = default_params
    special_params = 0
    n_params = len(params)
    for p, param in enumerate(params):
        if param == 'oxistates':
            special_params += 1
            n_params += 2
        elif param == 'econf':
            special_params += 1
            n_params += 4
        elif param == 'block':
            special_params += 1
        elif param == 'ionenergies':
            special_params += 1
    dat = []
    for Z in numbers:
        mnlv = get_mendeleev_params(Z, params=params)
        if special_params > 0:
            line = mnlv[:-special_params]
        else:
            line = mnlv
        for p, param in enumerate(params):
            if param == 'oxistates':
                line += [np.nanmin(mnlv[p]),
                         np.nanmedian(mnlv[p]),
                         np.nanmax(mnlv[p])]
            elif param == 'econf':
                line += list(n_outer(mnlv[p]))
            elif param == 'block':
                line += [float(block2number[mnlv[p]])]
            elif param == 'ionenergies':
                line += [mnlv[p]['1']]
        dat.append(line)
    if len(dat) == 0:
        dat.append([np.nan] * n_params)
    result = np.array(dat, dtype=float)
    if result.ndim == 1:
        result = result[np.newaxis]
    assert np.shape(result)[1] == n_params
    return result


def get_radius(z, params=['atomic_radius', 'covalent_radius_cordero']):
    """Return a metric of atomic radius.

    Parameters
    ----------
    z : int
        Atomic number.
    params : list
        Atomic radius metrics in order of preference. The first successful
        value will be returned.
    """
    p = get_mendeleev_params(z, params=params)
    for r in p:
        if r is not None:
            # Return atomic radius in AAngstrom.
            return float(r) / 100.
    # Return atomic radius in AAngstrom.
    return covalent_radii[z]


def default_catlearn_radius(z):
    """Return the default CatLearn covalent radius of element z.

    Parameters
    ----------
    z : int
        Atomic number.
    """
    if z == 1 or z == 6:
        r = covalent_radii[z]
    else:
        r = get_radius(z, params=['atomic_radius', 'covalent_radius_cordero'])
    # 15% bond streching is allowed.
    r *= 1.17
    return r
