"""Slab adsorbate fingerprint functions for machine learning."""
import numpy as np
from ase.data import ground_state_magnetic_moments as gs_magmom
from .periodic_table_data import (get_mendeleev_params,
                                  list_mendeleev_params,
                                  default_params, make_labels)
from .base import BaseGenerator, check_labels
from .adsorbate_fingerprint import extra_slab_params


default_chalcogenide_fingerprinters = ['formal_charges',
                                       'min_cation',
                                       'max_cation',
                                       'mean_cation',
                                       'sum_cation']


class ChalcogenideFingerprintGenerator(BaseGenerator):
    def __init__(self, **kwargs):
        """Class containing functions for fingerprint generation.

        Parameters
        ----------
        params : list
            An optional list of parameters upon which to generate features.
        cn_max : int
            Optional integer for maximum expected coordination number.
        ion_number : int
            Optional atomic number.
        ion_charge : int
            Optional formal charge of that element.
        """
        # Periodic table data.
        if not hasattr(self, 'params'):
            self.slab_params = kwargs.get('params')

        if self.slab_params is None:
            self.slab_params = default_params + extra_slab_params

        # Ion atomic number
        if not hasattr(self, 'ion_number'):
            self.ion_number = kwargs.get('ion_number')

        if self.ion_number is None:
            self.ion_number = 8

        # Ion charge
        if not hasattr(self, 'ion_charge'):
            self.ion_charge = kwargs.get('ion_charge')

        if self.ion_charge is None:
            self.ion_charge = -2

        super(ChalcogenideFingerprintGenerator, self).__init__(**kwargs)

    def formal_charges(self, atoms):
        """Return a fingerprint based on formal charges.

        Parameters
        ----------
            atoms : object
        """
        if atoms is None:
            return ['site_charge_av', 'site_charge_sum',
                    'site_excess', 'slab_excess', 'slab_transferred']
        else:
            slab = atoms.subsets['slab_atoms']
            if self.ion_number not in atoms.numbers[slab]:
                return [0.] * 5

            site = atoms.subsets['site_atoms']
            cm = atoms.connectivity
            anion_charges = np.zeros(len(atoms))
            for i, atom in enumerate(slab):
                if atoms.numbers[atom] == self.ion_number:
                    anion_charges[atom] = self.ion_charge
            transfer = cm * np.vstack(anion_charges)
            row_sums = transfer.sum(axis=1)
            shared = self.ion_charge * transfer / np.vstack(row_sums)
            cation_charges = -np.nansum(shared, axis=0)
            all_charges = anion_charges + cation_charges

            site_excess = 0
            slab_excess = 0
            transferred = 0
            for j, atom in enumerate(slab):
                charge = all_charges[atom]
                oxistates = get_mendeleev_params(atoms.numbers[atom],
                                                 ['oxistates'])[0]
                if charge in oxistates:
                    continue
                else:
                    oxi_index = np.argmin(np.abs(charge - np.array(oxistates)))
                    oxistate = oxistates[oxi_index]
                    transferred += abs(charge - oxistate)
                    slab_excess += charge - oxistate
                    if atom in site:
                        site_excess += charge - oxistate

            site_charge_av = np.nanmean(all_charges[site])
            site_charge_sum = np.nansum(all_charges[site])

            return [site_charge_av, site_charge_sum,
                    site_excess, slab_excess, transferred]

    def mean_cation(self, atoms=None):
        """Function that takes an atoms objects and returns a fingerprint
        vector with properties averaged over the surface metal atoms
        closest to an add atom.

        Parameters
        ----------
            atoms : object
        """
        labels = make_labels(self.slab_params, '', '_cation_av')
        labels.append('ground_state_magmom_cation_av')
        if atoms is None:
            return labels
        else:
            numbers = [atoms[j].number for j in atoms.subsets['cation_atoms']]
            dat = list_mendeleev_params(numbers, params=self.slab_params)
            result = list(np.nanmean(dat, axis=0))
            result += [np.nanmean([gs_magmom[z] for z in numbers])]
            check_labels(labels, result, atoms)
            return result

    def min_cation(self, atoms=None):
        """Function that takes an atoms objects and returns a fingerprint
        vector with properties averaged over the surface metal atoms
        closest to an add atom.

        Parameters
        ----------
            atoms : object
        """
        labels = make_labels(self.slab_params, '', '_cation_min')
        labels.append('ground_state_magmom_cation_min')
        if atoms is None:
            return labels
        else:
            numbers = [atoms[j].number for j in atoms.subsets['cation_atoms']]
            dat = list_mendeleev_params(numbers, params=self.slab_params)
            result = list(np.nanmin(dat, axis=0))
            result += [np.nanmin([gs_magmom[z] for z in numbers])]
            check_labels(labels, result, atoms)
            return result

    def max_cation(self, atoms=None):
        """Function that takes an atoms objects and returns a fingerprint
        vector with properties averaged over the surface metal atoms
        closest to an add atom.

        Parameters
        ----------
            atoms : object
        """
        labels = make_labels(self.slab_params, '', '_cation_max')
        labels.append('ground_state_magmom_cation_max')
        if atoms is None:
            return labels
        else:
            numbers = [atoms[j].number for j in atoms.subsets['cation_atoms']]
            dat = list_mendeleev_params(numbers, params=self.slab_params)
            result = list(np.nanmax(dat, axis=0))
            result += [np.nanmax([gs_magmom[z] for z in numbers])]
            check_labels(labels, result, atoms)
            return result

    def median_cation(self, atoms=None):
        """Function that takes an atoms objects and returns a fingerprint
        vector with properties averaged over the surface metal atoms
        closest to an add atom.

        Parameters
        ----------
            atoms : object
        """
        labels = make_labels(self.slab_params, '', '_cation_med')
        labels.append('ground_state_magmom_cation_med')
        if atoms is None:
            return labels
        else:
            numbers = [atoms[j].number for j in atoms.subsets['cation_atoms']]
            dat = list_mendeleev_params(numbers, params=self.slab_params)
            result = list(np.nanmedian(dat, axis=0))
            result += [np.nanmedian([gs_magmom[z] for z in numbers])]
            check_labels(labels, result, atoms)
            return result

    def sum_cation(self, atoms=None):
        """Function that takes an atoms objects and returns a fingerprint
        vector with properties summed over the surface metal atoms
        closest to an add atom.

        Parameters
        ----------
            atoms : object
        """
        labels = make_labels(self.slab_params, '', '_cation_sum')
        labels.append('ground_state_magmom_cation_sum')
        if atoms is None:
            return labels
        else:
            numbers = [atoms[j].number for j in atoms.subsets['cation_atoms']]
            dat = list_mendeleev_params(numbers, params=self.slab_params)
            result = list(np.nansum(dat, axis=0))
            result += [np.nansum([gs_magmom[z] for z in numbers])]
            check_labels(labels, result, atoms)
            return result
