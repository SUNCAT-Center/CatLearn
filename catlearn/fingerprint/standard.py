"""Standard fingerprint functions.

These feature sets should perform relatively well on a variety of different
systems. They are general descriptors based predominantly on the elemental
properties and in some cases structure.

This class inherits from the catlearn.fingerprint.BaseGenerator function.
"""
from __future__ import absolute_import
from __future__ import division

import json
import numpy as np
import warnings

from ase.data import chemical_symbols

from catlearn import __path__ as catlearn_path
from catlearn.featurize.base import BaseGenerator


default_molecule_fingerprinters = [
                                   'element_parameter_vec',
                                   'eigenspectrum_vec',
                                   'composition_vec',
                                   'distance_vec',
                                   'bag_elements'
                                   'bag_edges',
                                   'bag_element_cn'
                                   ]


class StandardFingerprintGenerator(BaseGenerator):
    """Function to build a fingerprint vector based on an atoms object."""

    def __init__(self, **kwargs):
        """Standard fingerprint generator setup.

        Parameters
        ----------
        atom_types : list
            Unique atomic types in the systems. Types are denoted by atomic
            number e.g. for CH4 set [1, 6].
        atom_len : int
            The maximum length of all atomic systems that will be passed in a
            data set.
        element_parameters : str, list
            Optional variable to be passed if element_parameter_vec is to be
            called. Type of atomic parameter upon which to compile the feature
            vector. A full list of atomic parameters can be found here:
            https://pypi.python.org/pypi/mendeleev/
        """
        if not hasattr(self, 'atom_types'):
            self.atom_types = kwargs.get('atom_types')
        if not hasattr(self, 'atom_len'):
            self.atom_len = kwargs.get('atom_len')
        if not hasattr(self, 'element_parameters'):
            self.element_parameters = kwargs.get('element_parameters')

        if not hasattr(self, 'element_data'):
            # Load the Mendeleev parameter data into memory
            with open('/'.join(catlearn_path[0].split('/')[:-1]) +
                      '/catlearn/data/proxy-mendeleev.json') as f:
                self.element_data = json.load(f)

        # Coordination number bounds.
        if not hasattr(self, 'cn_max'):
            self.cn_max = kwargs.get('cn_max')
        if self.cn_max is None:
            self.cn_max = 4

        if not hasattr(self, 'cn_min'):
            self.cn_min = kwargs.get('cn_min')
        if self.cn_min is None:
            self.cn_min = 0

        super(StandardFingerprintGenerator, self).__init__(**kwargs)

    def composition_vec(self, data):
        """Function to return a feature vector based on the composition.

        Parameters
        ----------
        data : object
            Data object with atomic numbers available.

        Returns
        -------
        features : array
            Vector containing a count of the different atomic types, e.g. for
            CH3OH the vector [1, 4, 1] would be returned.
        """
        # Return feature names in no atomic data is passed.
        if data is None:
            msg = 'Class must have atom_types set to return feature names.'
            assert hasattr(self, 'atom_types') and self.atom_types is not \
                None, msg
            return ['{}_count'.format(n) for n in self.atom_types]

        ano = self.get_atomic_numbers(data)

        # WARNING: Will be set permanently whichever atom is first passed.
        if self.atom_types is None:
            msg = 'atom_types variable will be set permanently to whichever '
            msg += 'atom object is first passed'
            warnings.warn(msg)
            self.atom_types = sorted(frozenset(ano))

        return np.array([list(ano).count(sym) for sym in self.atom_types])

    def element_parameter_vec(self, data):
        """Function to return a vector based on a defined paramter.

        The vector is compiled based on the summed parameters for each
        elemental type as well as the sum for all atoms.

        Parameters
        ----------
        data : object
            Data object with atomic numbers available.

        Returns
        -------
        features : array
            An n + 1 array where n in the length of self.atom_types.
        """
        msg = 'The variable element_parameters must be set in the feature '
        msg += 'generator class.'
        assert self.element_parameters is not None, msg

        if not isinstance(self.element_parameters, list):
            self.element_parameters = [self.element_parameters]

        # Return feature names in no atomic data is passed.
        if data is None:
            msg = 'Class must have atom_types set to return feature names.'
            assert hasattr(self, 'atom_types') and self.atom_types is not \
                None, msg
            names = []
            for p in self.element_parameters:
                names += ['sum_{0}_{1}'.format(n, p) for n in self.atom_types]
                names += ['sum_all_{0}'.format(p), 'mean_all_{0}'.format(p)]
            return names

        # Get the composition data.
        comp = self.composition_vec(data)

        # Generate the actual features.
        features = np.asarray([])
        for p in self.element_parameters:
            plist = [self.element_data[str(an)].get(p) for an in
                     self.atom_types]
            # Converts None type to np.nan.
            plist = np.asarray(plist, dtype=np.float32)

            f = np.zeros(len(comp) + 2)
            f[:len(comp)] = np.multiply(comp, plist)
            var = [np.nansum(f), np.nanmean(f)]
            f[len(comp):] = var
            features = np.concatenate((features, f))

        return features

    def element_mass_vec(self, data):
            """Function to return a vector based on mass parameter.

            Parameters
            ----------
            data : object
                Data object with atomic masses available.

            Returns
            -------
            features : ndarray
                Vector of the summed mass.
            """
            # Return feature names in no atomic data is passed.
            if data is None:
                return ['sum_mass']
            # Return the summed mass of the atoms object.
            return np.array([sum(self.get_masses(data))])

    def _get_coulomb(self, data):
        """Generate the coulomb matrix.

        A more detailed discussion of the coulomb features can be found here:
        https://doi.org/10.1103/PhysRevLett.108.058301

        Parameters
        ----------
        data : object
            Data object with Cartesian coordinates and atomic numbers
            available.

        Returns
        -------
        coulomb : ndarray
            The coulomb matrix, (n, n) atoms in size.
        """
        if len(data) < 2:
            raise ValueError(
                'Columb matrix requires atoms object with at least 2 atoms')

        dm = self.get_all_distances(data)
        np.fill_diagonal(dm, 1)

        # Make coulomb matrix
        ano = self.get_atomic_numbers(data)
        coulomb = np.outer(ano, ano) / dm

        diagonal = 0.5 * ano ** 2.4
        np.fill_diagonal(coulomb, diagonal)

        return coulomb

    def eigenspectrum_vec(self, data):
        """Sorted eigenspectrum of the Coulomb matrix.

        Parameters
        ----------
        data : object
            Data object with Cartesian coordinates and atomic numbers
            available.

        Returns
        -------
        features : ndarray
            Sorted Eigen values of the coulomb matrix, n atoms is size.
        """
        # Return feature names in no atomic data is passed.
        if data is None:
            msg = 'Class must have atom_len set to return feature names.'
            assert hasattr(self, 'atom_len') and self.atom_len is not \
                None, msg
            return ['eig_{}'.format(n) for n in range(self.atom_len)]

        features = np.zeros(self.atom_len)
        coulomb = self._get_coulomb(data)

        v = np.linalg.eigvals(coulomb)
        v[::-1].sort()
        features[:len(v)] = v

        return features

    def distance_vec(self, data):
        """Averaged distance between e.g. A-A atomic pairs.

        Parameters
        ----------
        data : object
            Data object with Cartesian coordinates and atomic numbers
            available.

        Returns
        -------
        features : ndarray
            Vector of averaged distances between homoatomic atoms.
        """
        # Return feature names in no atomic data is passed.
        if data is None:
            msg = 'Class must have atom_types set to return feature names.'
            assert hasattr(self, 'atom_types') and self.atom_types is not \
                None, msg
            return ['{0}-{0}_dist'.format(n) for n in self.atom_types]

        features = []
        an = self.get_atomic_numbers(data)
        pos = self.get_positions(data)

        # Get unique atom types.
        if self.atom_types is None:
            msg = 'atom_types variable will be set permanently to whichever '
            msg += 'atom object is first passed'
            warnings.warn(msg)
            self.atom_types = sorted(frozenset(an))

        for at in self.atom_types:
            ad = 0.
            co = 0.
            for i, j in zip(an, pos):
                if i == at:
                    for k, l in zip(an, pos):
                        if k == at and all(j != l):
                            co += 1.
                            ad += np.linalg.norm(j - l)
            if co != 0:
                co /= ad
            features.append(co)
        return features

    def bag_elements(self, atoms):
        """Returns the bag of elements, defined as counting occurence of
        elements in a given structure.
        This is mostly useful for subtracting atomization energies.

        Parameters
        ----------
        atoms : object

        Returns
        ----------
        features : list
        """
        # range of element types.
        labels = ['bag_' + chemical_symbols[z] for z in self.atom_types]
        if atoms is None:
            return labels
        else:
            # empty bag atoms.
            bag = np.zeros(len(labels))
            for i, z in enumerate(self.atom_types):
                bag[i] += list(atoms.numbers).count(z)

            return list(bag)

    def bag_edges(self, atoms):
        """Returns the bag of connections, defined as counting connections
        between types of elements pairs. We define the bag as a vector, e.g.
        return [Number of C-H connections, # C-C, # C-O, ..., # M-X]

        Parameters
        ----------
        atoms : object

        Returns
        ----------
        features : list
        """
        # range of element types
        n_elements = len(self.atom_types)
        if atoms is None:
            symbols = np.array([chemical_symbols[z] for z in self.atom_types])
            rows, cols = np.meshgrid(symbols, symbols)
            pairs = np.core.defchararray.add(rows, cols)
            labels = ['bag_' + c for c in pairs[np.triu_indices_from(pairs)]]
            return labels
        else:
            # empty bag of bond types.
            boc = np.zeros([n_elements, n_elements])

            natoms = len(atoms)
            cm = np.array(atoms.connectivity)
            np.fill_diagonal(cm, 0)

            bonds = np.where(np.ravel(np.triu(cm)) > 0)[0]
            for b in bonds:
                # Get bonded atomic numbers.
                z_row, z_col = np.unravel_index(b, [natoms, natoms])
                bond_index = sorted((atoms.numbers[z_row],
                                     atoms.numbers[z_col]))
                bond_type = tuple((self.atom_types.index(bond_index[0]),
                                   self.atom_types.index(bond_index[1])))
                # Count bonds in upper triangle.
                boc[bond_type] += 1
            return boc[np.triu_indices_from(boc)].tolist()

    def bag_element_cn(self, atoms):
        """Bag elements folded with coordination numbers,
        e.g. number of C with CN = 4, number of C with CN = 3, ect.

        Parameters
        ----------
        atoms : object
            ASE Atoms object.

        Returns
        ----------
        features : list
            If None was passed, the elements are strings, naming the feature.
        """
        labels = []
        atom_symbols = [chemical_symbols[z] for z in self.atom_types]
        index_symbols = {}
        for j, s in enumerate(atom_symbols):
            index_symbols[s] = j
            labels += ['bag_cn_' + s + '_' + str(n) for
                       n in range(self.cn_min, self.cn_max+1)]
        if atoms is None:
            return labels
        else:
            s_cn_matrix = np.zeros([len(self.atom_types),
                                    self.cn_max+1-self.cn_min])
            cm = np.array(atoms.connectivity, dtype=int)
            for i, atom in enumerate(atoms):
                cn = cm[i, :].sum()
                cn_i = cn - self.cn_min
                if cn > self.cn_max or cn < self.cn_min:
                    print(atoms.info['key_value_pairs'], cn)
                    warnings.warn('Coordination number out of bounds')
                    return [np.nan] * len(labels)
                s_cn_matrix[index_symbols[atoms.symbols[i]], cn_i] += 1
            fingerprint = list(np.ravel(s_cn_matrix))
            return fingerprint

    def bag_edges_cn(self, atoms):
        """Returns the bag of connections folded with coordination numbers of
        the node atoms.

        Parameters
        ----------
        atoms : object

        Returns
        ----------
        features : list
        """
        # range of element types
        atom_symbols = [chemical_symbols[z] for z in self.atom_types]
        nodes = []
        for j, s in enumerate(atom_symbols):
            nodes += [s + str(n) for n in
                      range(self.cn_min, self.cn_max+1)]
        if atoms is None:
            rows, cols = np.meshgrid(nodes, nodes)
            pairs = np.core.defchararray.add(rows, cols)
            labels = ['bag_' + c for c in pairs[np.triu_indices_from(pairs)]]
            return labels
        else:
            # empty bag of bond types.
            n_elements_cn = len(self.atom_types) * \
                (self.cn_max - self.cn_min + 1)
            boc = np.zeros([n_elements_cn, n_elements_cn], dtype=int)

            natoms = len(atoms)
            cm = np.array(atoms.connectivity, dtype=int)
            np.fill_diagonal(cm, 0)
            cn_list = cm.sum(axis=1)

            bonds = np.where(np.ravel(np.triu(cm)) > 0)[0]
            for b in bonds:
                # Get bonded atomic indices.
                i_row, i_col = np.unravel_index(b, [natoms, natoms])
                z = (atoms.numbers[i_row], atoms.numbers[i_col])
                cn = (cn_list[i_row], cn_list[i_col])
                bond_index = np.lexsort((cn, z))
                node_a = chemical_symbols[np.array(z)[bond_index[0]]] + \
                    str(np.array(cn)[bond_index[0]])
                node_b = chemical_symbols[np.array(z)[bond_index[1]]] + \
                    str(np.array(cn)[bond_index[1]])

                # Get bond types.
                bond_type = tuple((nodes.index(node_a),
                                   nodes.index(node_b)))
                # Count bonds in upper triangle.
                boc[bond_type] += 1
            return boc[np.triu_indices_from(boc)].tolist()
