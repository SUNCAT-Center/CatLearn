"""Function pulling atomic data for elements."""
import json
from atoml import __path__ as atoml_path
from ase.data import ground_state_magnetic_moments
import numpy as np

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
    ns = 0
    np = 0
    nd = 0
    nf = 0
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
            ns += n_shell
        elif 'p' in shell:
            np += n_shell
        elif 'd' in shell:
            nd += n_shell
        elif 'f' in shell:
            nf += n_shell
    return n_tot, ns, np, nd, nf

def get_mendeleev_params(atomic_number, params=None, extra_params=[]):
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
        params = list(default_params)

    if extra_params:
        params += extra_params

    var = []
    for an in atomic_number:
        for _ in params:
            var += [data[str(an)].get(_)]

    return var

def average_mendeleev_params(numbers, params=None):
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
        params=list(default_params)
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
    try:
        result = list(np.nanmean(dat, axis=0, dtype=float))
    except:
        print(dat)
    return result

def sum_mendeleev_params(numbers, params=None):
    """Returns a list of summed parameters weighted to the frequecy of
    occurence in a list of atomic numbers
    
    Parameters
    ----------
        numbers : list
            atomic numbers.
        params : list
            elemental parameters.
    """
    if params is None:
        params=list(default_params)
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
    try:
        result = list(np.nansum(dat, axis=0, dtype=float))
    except:
        print(dat)
    return result