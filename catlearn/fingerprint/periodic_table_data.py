"""Function pulling atomic data for elements.

This is typically used in conjunction with other fiungerprint generators to
combine general atomic data with more specific properties.
"""
import warnings
import json
from catlearn import __path__ as catlearn_path
from ase.data import covalent_radii, atomic_numbers
import numpy as np
import re


# Load the Mendeleev parameter data into memory
with open('/'.join(catlearn_path[0].split('/')[:-1]) +
          '/catlearn/data/proxy-mendeleev.json') as f:
    data = json.load(f)


block2number = {'s': [1, 0, 0, 0],
                'p': [0, 1, 0, 0],
                'd': [0, 0, 1, 0],
                'f': [0, 0, 0, 1]}


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
            n_params += 3
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
                line += block2number[mnlv[p]]
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


def stat_mendeleev_params(composition, params=None):
    """Return an n by p array, containing p parameters of n atoms and
    stoichiometry weigths associated with the unique elements in the formula.

    Parameters
    ----------
        composition : str
            chemical composition formula. Floats are accepted.
        params : list
            elemental parameters.
    """
    coefs = [float(s) for s in re.findall(r"[-+]?\d*\.\d+|\d+", composition)]

    numbers = []
    weigths = []

    symbol = ''
    n_float = -1
    n_symbol = -1
    for n_char, char in enumerate(composition):
        if char.isupper():
            if n_symbol >= 0:
                if n_float < n_symbol:
                    # Append 1 if no previous coefficient.
                    n_float += 1
                    weigths.append(1.)
                # append previous Z
                try:
                    numbers.append(atomic_numbers[symbol])
                except KeyError:
                    n_symbol -= 1
                    weigths.pop(-1)
            # Start new symbol.
            symbol = char
            n_symbol += 1
            scanning_float = False
        elif char.islower():
            # Continue symbol.
            if char == 'x':
                weigths.append(np.nan)
                n_float += 1
            else:
                symbol += char
        elif char.isdigit():
            if n_symbol == -1:
                warnings.warn(composition + ' skipped')
                continue
            # Retrive coefficient.
            if scanning_float:
                pass
            else:
                weigths.append(coefs[n_float])
                n_float += 1
                scanning_float = True
        else:
            warnings.warn(composition + ' skipped')
            continue

    if n_float < n_symbol:
        # Append 1 if no previous coefficient.
        n_float += 1
        weigths.append(1.)
    # Append last symbol.
    try:
        numbers.append(atomic_numbers[symbol])
    except KeyError:
        n_float -= 1
        weigths.pop(-1)
    assert len(weigths) == len(numbers), composition

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
            n_params += 3
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
                line += block2number[mnlv[p]]
            elif param == 'ionenergies':
                line += [mnlv[p]['1']]
        dat.append(line)
    if len(dat) == 0:
        dat.append([np.nan] * n_params)
    result = np.array(dat, dtype=float)
    if result.ndim == 1:
        result = result[np.newaxis]
    assert np.shape(result)[1] == n_params
    return result, weigths


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
    if z == 6:
        radius = covalent_radii[z]
    elif z == 1:
        radius = covalent_radii[z] + 0.2
    else:
        radius = get_radius(z)
    # Some bond streching is allowed.
    return radius * 1.15 + 0.1


def make_labels(params, prefix, suffix):
    """Return a list of feature labels.

    Parameters
    ----------
    params : list
        Parameter keys.
    prefix : str
        Appended in front of each parameter key.
    suffix : str
        Appended to end of each parameter key.

    Returns
    ----------
    labels : list
    """
    labels = []
    for p in params:
        if p == 'oxistates':
            labels += [prefix + 'oxi_' + s + suffix for
                       s in ['min', 'med', 'max']]
        elif p == 'block':
            labels += [prefix + s + 'block' + suffix for
                       s in ['s', 'p', 'd', 'f']]
        elif p == 'econf':
            labels += [prefix + 'ne_' + s + suffix for
                       s in ['outer', 's', 'p', 'd', 'f']]
        elif p == 'ionenergies':
            labels.append(prefix + 'ionenergy' + suffix)
        else:
            labels.append(prefix + p + suffix)
    return labels
