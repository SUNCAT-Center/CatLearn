#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pair distribution function.
"""
import numpy as np


def pair_distribution(images, bins=101, bounds=None, mic=True, element=None):
    """Return the pair distribution function from a list of atoms objects.

    Parameters
    ----------
    images : list
        List of atoms objects.
    bins : int
        Number of bins
    bounds : tuple
        Optional upper and lower bound of distances.
    mic : boolean
        Use minimum image convention. Set to False for non-periodic structures.
    subset : list
        Optionally select a subset of atomic indices to include.
    """
    if bounds is None:
        bounds = (0.3, 3.)

    pdf, x0 = np.histogram([], bins=bins, range=bounds)
    n = 0
    for atoms in images:
        dist, x = _distance_hist(atoms, bins=bins, bounds=bounds,
                                 mic=mic, element=element)
        assert np.allclose(x, x0)
        pdf = np.nansum([pdf, dist], axis=0)  # np.add(pdf, dist)
        n += 1

    # Normalize to volume and number.
    pdf = pdf / (n * 4 * np.pi * (x0[1:] ** 3. - x0[:-1] ** 3) / 3.0)

    # Center bins.
    dx = (max(x0) - min(x0)) / len(x0)
    return pdf, x0[:-1] + dx/2.


def pair_deviation(images, cutoffs, bins=33,
                   bounds=None, mic=True, element=None):
    """Return distribution of deviations from atom-pair nominal bond length.

    Parameters
    ----------
    images : list
        List of atoms objects.
    cutoffs : dictionary
        Subtract elemental cutoff radii from distances.
        This is a useful for testing cutoff radii.
    bins : int
        Number of bins
    bounds : tuple
        Optional upper and lower bound of distances.
    mic : boolean
        Use minimum image convention. Set to False for non-periodic structures.
    subset : list
        Optionally select a subset of atomic indices to include.
    """
    if bounds is None:
        bounds = (-1., 1.)

    pdf, x0 = np.histogram([], bins=bins, range=bounds)
    n = 0
    for atoms in images:
        dist, x = _distance_hist(atoms, bins=bins, bounds=bounds,
                                 mic=mic,
                                 element=element,
                                 subtract_cutoffs=cutoffs)
        assert np.allclose(x, x0)
        pdf = np.nansum([pdf, dist], axis=0)  # np.add(pdf, dist)
        n += 1

    # Center bins and normalize to number.
    dx = (max(x0) - min(x0)) / len(x0)
    return pdf / n, x0[:-1] + dx/2.


def _distance_hist(atoms, bins, bounds, mic=True, element=None,
                   subtract_cutoffs=False):
    """Return a histogram of interatomic distances from an atoms object.

    Parameters
    ----------
    atoms : object
        atoms object.
    bins : int
        Number of bins
    bounds : tuple
        Upper and lower bound of distances.
    mic : boolean
        Use minimum image convention. Set to False for non-periodic structures.
    element : tuple or int
        Optionally select a subset of atomic indices to include.
    subtract_cutoffs : dictionary
        Optionally subtract elemental cutoff radii from distances.
        This is a useful for testing cutoff radii.
    """
    if isinstance(element, tuple) and element[0] == element[1]:
        subset = [i for i, z in enumerate(atoms.numbers) if z == element[0]]
        atoms = atoms[subset].copy()
    d = atoms.get_all_distances(atoms)

    if subtract_cutoffs is not False:
        if isinstance(subtract_cutoffs, dict):
            cutoffs = [subtract_cutoffs[z] for z in atoms.numbers]
        elif (isinstance(subtract_cutoffs, list) and
              len(subtract_cutoffs) == len(atoms)):
            cutoffs = subtract_cutoffs
        d = d - np.ravel(cutoffs)
        d = d - np.vstack(cutoffs)

    if isinstance(element, int):
        subset = [i for i, z in enumerate(atoms.numbers) if z == element]
        d = d[subset, :]
    elif isinstance(element, tuple) and element[0] != element[1]:
        subset_a = [i for i, z in enumerate(atoms.numbers) if z == element[0]]
        subset_b = [i for i, z in enumerate(atoms.numbers) if z == element[1]]
        d = d[subset_a, :][:, subset_b]

    n = 0
    dist, x0 = np.histogram([], bins=bins, range=bounds)
    for atom, row in enumerate(d):
        h, x = np.histogram(row, bins=bins, range=bounds)
        assert np.allclose(x, x0)
        dist = np.add(dist, h)
        n += 1

    # Normalize to number.
    return dist / n, x0
