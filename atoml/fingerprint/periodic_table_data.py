"""Function pulling atomic data for elements."""
import json
from atoml import __path__ as atoml_path

# Load the Mendeleev parameter data into memory
with open('/'.join(atoml_path[0].split('/')[:-1]) +
          '/atoml/data/proxy-mendeleev.json') as f:
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
    """Returns a list of average parameters weighted to the frequecy of
    occurence in a list of atomic numbers

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
    for p, param in enumerate(params):
        if param == 'econf':
            special_params += 1
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
            if param == 'econf':
                line += list(n_outer(mnlv[p]))
            elif param == 'block':
                line += [float(block2number[mnlv[p]])]
            elif param == 'ionenergies':
                line += [mnlv[p]['1']]
        dat.append(line)
    return dat


def get_radius(z):
    p = get_mendeleev_params(z, params=['atomic_radius',
                                        'covalent_radius_cordero'])
    if p[0] is not None:
        r = p[0]
    elif p[1] is not None:
        r = p[1]
    return float(r) / 100.
