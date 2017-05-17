""" Particle fingerprint functions generating sets of features with a physical
    meaning.
"""
from __future__ import absolute_import
from __future__ import division

import numpy as np
from itertools import product

from ase.ga.utilities import (get_nnmat, get_nndist, get_atoms_distribution,
                              get_neighborlist, get_atoms_connections, get_rdf)
no_asap = False
try:
    from asap3.analysis.rdf import RadialDistributionFunction
except ImportError:
    no_asap = True


class ParticleFingerprintGenerator(object):
    def __init__(self, atom_numbers=None, max_bonds=13, get_nl=False, dx=0.2,
                 cell_size=50., nbin=4, rmax=8., nbins=5):
        """ atom_numbers: list
                List of unique atomic numbers. Supplying the list can remove
                problems with different vectors being generated if the
                composition varies.

            max_bonds: int
                Count up to the specified number of bonds. Default is 0 to 12.

            get_nl: boolean
                Specify whether to recalculate the neighborlist. Default is
                False.

            dx: float
                Cutoff added to the covalent radii to calculate the
                neighborlist.
        """
        self.atom_numbers = atom_numbers
        self.max_bonds = max_bonds
        self.get_nl = get_nl
        self.dx = dx
        self.cell_size = cell_size
        self.nbin = nbin
        self.nbins = nbins
        self.rmax = rmax

    def nearestneighbour_fpv(self, atoms):
        """ Function that takes an atoms objects and returns a fingerprint
            vector based on the nearest neighbour average as described in
            Topics in Catalysis, 2014, 57, 33-39.
        """
        return get_nnmat(atoms)

    def bond_count_fpv(self, atoms):
        """ Function that takes a list of atoms objects and returns a list of
            fingerprint vectors based on more detailed bond counting than the
            nearestneighbour_fpv. This function also includes a distribution
            measure as well as accounting for coordination number.
        """
        if self.get_nl:
            # Define the neighborlist.
            atoms.info['data']['neighborlist'] = get_neighborlist(atoms,
                                                                  dx=self.dx)

        elements = sorted(frozenset(atoms.get_chemical_symbols()))

        # Get coordination number counting.
        dm = atoms.get_all_distances()
        nndist = get_nndist(atoms, dm) + 0.2
        track_nnmat = np.zeros((self.max_bonds, len(elements), len(elements)))
        for j in range(len(atoms)):
            row = elements.index(atoms[j].symbol)
            neighbors = [k for k in range(len(dm[j]))
                         if 0.1 < dm[j][k] < nndist]
            ln = len(neighbors)
            if ln > 12:
                continue
            for l in neighbors:
                column = elements.index(atoms[l].symbol)
                track_nnmat[ln][row][column] += 1

        return track_nnmat.ravel()

    def distribution_fpv(self, atoms):
        """ Function that takes a list of atoms objects and returns a list of
            fingerprint vectors based on a atomic distribution measure.

            cell_size: float
                Some of the utility functions work better when the cell is set
                and atoms centered. The size of the cell can be set with this
                variable. Default 50.0 angstroms.

            nbin: int
                The number of bins supplied to the get_atoms_distribution
                function.
        """
        # Center the atoms, works better for some utility functions.
        atoms.set_cell([self.cell_size, self.cell_size, self.cell_size])
        atoms.center()

        if self.get_nl:
            # Define the neighborlist.
            atoms.info['data']['neighborlist'] = get_neighborlist(atoms,
                                                                  dx=self.dx)

        # If unique atomic numbers not supplied. Generate it now.
        if self.atom_numbers is None:
            self.atom_numbers = frozenset(atoms.get_atomic_numbers())
        # Get the atomic distribution of each atom type.
        dist = []
        for i in self.atom_numbers:
            dist += get_atoms_distribution(atoms, number_of_bins=self.nbin,
                                           no_count_types=[i])
        return dist

    def connections_fpv(self, atoms):
        """ Function that takes a list of atoms objects and returns a list of
            fingerprint vectors based on the sum of the numbers of atoms with
            a certain number of connections.
        """
        if self.get_nl:
            # Define the neighborlist.
            atoms.info['data']['neighborlist'] = get_neighborlist(atoms,
                                                                  dx=self.dx)

        fp = []
        if self.atom_numbers is None:
            # Get unique atom types.
            self.atom_numbers = frozenset(atoms.get_atomic_numbers())
        for an in self.atom_numbers:
            conn = get_atoms_connections(atoms, max_conn=self.max_bonds,
                                         no_count_types=[an])
            for i in conn:
                fp.append(i)

        return fp

    def rdf_fpv(self, atoms):
        """Return a list of concatenated partial rdfs
        for use as fingerprint vector."""

        if not no_asap:
            rf = RadialDistributionFunction(atoms,
                                            rMax=self.rmax,
                                            nBins=self.nbins).get_rdf
            kwargs = {}
        else:
            rf = get_rdf
            dm = atoms.get_all_distances()
            kwargs = {'atoms': atoms, 'rmax': self.rmax,
                      'nbins': self.nbins, 'no_dists': True,
                      'distance_matrix': dm}

        fp = []
        for c in product(set(atoms.numbers), repeat=2):
            fp.extend(rf(elements=c, **kwargs))

        return fp
