"""Nanoparticle fingerprint functions.

These functions will typically perform well at describing chemical ordering
within alloyed nanoparticles. However, they may be applicable to other
applications where bond counting or coordination numbers are important
descriptors.

This class inherits from the catlearn.fingerprint.BaseGenerator function.
"""
from __future__ import absolute_import
from __future__ import division

import numpy as np
from itertools import product
import warnings

from ase.ga.utilities import (get_atoms_distribution, get_atoms_connections,
                              get_rdf)

from .base import BaseGenerator

no_asap = False
try:
    from asap3.analysis.rdf import RadialDistributionFunction
except ImportError:
    no_asap = True


default_particle_fingerprinters = [
                                   'nearestneighbour_vec'
                                   ]


class ParticleFingerprintGenerator(BaseGenerator):
    """Function to build a fingerprint vector based on an atoms object."""

    def __init__(self, **kwargs):
        """Particle fingerprint generator setup.

        Parameters
        ----------
        atom_types : list
            List of unique atomic numbers.
        max_bonds : int
            Count up to the specified number of bonds. Default is 0 to 12.
        get_nl : boolean
            Specify whether to recalculate the neighborlist. Default is False.
        dx : float
            Cutoff added to the covalent radii to calculate the neighborlist.
        cell_size : float
            Set unit cell size, default is 50.0 angstroms.
        nbin : int
            The number of bins supplied for distribution functions.
        """
        if not hasattr(self, 'atom_types'):
            self.atom_types = kwargs.get('atom_types')

        self.max_bonds = kwargs.get('max_bonds', 13)
        self.get_nl = kwargs.get('get_nl', False)
        self.dx = kwargs.get('dx', 0.2)
        self.cell_size = kwargs.get('cell_size', 50.)
        self.nbin = kwargs.get('nbin', 5)
        self.rmax = kwargs.get('rmax', 8.)

        super(ParticleFingerprintGenerator, self).__init__(**kwargs)

    def nearestneighbour_vec(self, data):
        """Nearest neighbour average, Topics in Catalysis, 2014, 57, 33.

        This is a slightly modified version of the code found in the `ase.ga`
        module.

        Parameters
        ----------
        data : object
            Data object with atomic numbers available.

        Returns
        -------
        nnlist : list
            Feature vector that will be n**2 where n is the number of atomic
            species passed to the class.
        """
        # Return feature names in no atomic data is passed.
        if data is None:
            msg = 'Class must have atom_types set to return feature names.'
            assert hasattr(self, 'atom_types') and self.atom_types is not \
                None, msg
            names = []
            for i in self.atom_types:
                names += ['{0}_{1}_nnmat'.format(i, j)
                          for j in self.atom_types]
            return names

        # WARNING: Will be set permanently whichever atom is first passed.
        if self.atom_types is None:
            msg = 'atom_types variable will be set permanently to whichever '
            msg += 'atom object is first passed'
            warnings.warn(msg)
            self.atom_types = sorted(frozenset(self.get_atomic_numbers(data)))

        # Calculate the distance parameters.
        dm = self.get_all_distances(data)
        rdf, dists = get_rdf(data, 10., 200, dm)
        nndist = dists[np.argmax(rdf)] + 0.2

        # initialize correct shape feature martix.
        nnmat = np.zeros((len(self.atom_types), len(self.atom_types)))

        # Calculate the neighbor properties.
        for i, d in enumerate(data):
            row = [j for j in range(len(self.atom_types))
                   if d.number == self.atom_types[j]][0]
            neighbors = [j for j in range(len(dm[i])) if dm[i][j] < nndist]
            for n in neighbors:
                column = [j for j in range(len(self.atom_types))
                          if data[n].number == self.atom_types[j]][0]
                nnmat[row][column] += 1

        # Normalize the features.
        for i, el in enumerate(self.atom_types):
            nnmat[i] /= len([j for j in range(len(data))
                             if data[int(j)].number == el])

        # convert matrix to vector and replace np.nan values.
        nnlist = np.nan_to_num(nnmat.flatten())

        return nnlist

    def bond_count_vec(self, data):
        """Bond counting with a distribution measure for coordination.

        Parameters
        ----------
        data : object
            Data object with atomic distances.

        Returns
        -------
        track_nnmat : list
            List with summed number of atoms with given coordination numbers.
        """
        elements = sorted(set(self.get_atomic_numbers(data)))

        # Get coordination number counting.
        dm = self.get_all_distances(data)
        rdf, dists = get_rdf(data, 10., 200, dm)
        nndist = dists[np.argmax(rdf)] + 0.2
        track_nnmat = np.zeros((self.max_bonds, len(elements), len(elements)))
        for j in range(len(data)):
            row = elements.index(data[j].number)
            neighbors = [k for k in range(len(dm[j]))
                         if 0.1 < dm[j][k] < nndist]
            ln = len(neighbors)
            if ln > 12:
                continue
            for l in neighbors:
                column = elements.index(data[l].number)
                track_nnmat[ln][row][column] += 1

        return track_nnmat.ravel()

    def distribution_vec(self, data):
        """Return atomic distribution measure."""
        # Center the atoms, works better for some utility functions.
        data.set_cell([self.cell_size, self.cell_size, self.cell_size])
        data.center()

        # WARNING: Will be set permanently whichever atom is first passed.
        if self.atom_types is None:
            msg = 'atom_types variable will be set permanently to whichever '
            msg += 'atom object is first passed'
            warnings.warn(msg)
            self.atom_types = sorted(frozenset(self.get_atomic_numbers(data)))

        # Get the atomic distribution of each atom type.
        dist = []
        for i in self.atom_types:
            dist += get_atoms_distribution(data, number_of_bins=self.nbin,
                                           no_count_types=[i])
        return dist

    def connections_vec(self, data):
        """Sum atoms with a certain number of connections."""
        # WARNING: Will be set permanently whichever atom is first passed.
        if self.atom_types is None:
            msg = 'atom_types variable will be set permanently to whichever '
            msg += 'atom object is first passed'
            warnings.warn(msg)
            self.atom_types = sorted(frozenset(self.get_atomic_numbers(data)))

        fp = []
        for an in self.atom_types:
            conn = get_atoms_connections(data, max_conn=self.max_bonds,
                                         no_count_types=[an])
            for i in conn:
                fp.append(i)

        return fp

    def rdf_vec(self, data):
        """Return list of partial rdfs for use as fingerprint vector."""
        if not no_asap:
            rf = RadialDistributionFunction(
                data, rMax=self.rmax, nBins=self.nbin).get_rdf
            kwargs = {}
        else:
            rf = get_rdf
            dm = self.get_all_distances(data)
            kwargs = {
                'atoms': data, 'rmax': self.rmax, 'nbins': self.nbin,
                'no_dists': True, 'distance_matrix': dm}

        fp = []
        for c in product(set(data.numbers), repeat=2):
            fp.extend(rf(elements=c, **kwargs))

        return fp
