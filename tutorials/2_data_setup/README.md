# Data setup

This tutorial shows how to set up your training data. All prediction functions
accept training data in the form of a N x D matrix, where N is a number of
training examples and D is the number of descriptors. Each row in the matrix,
we call the fingerprint of a training example. Each column, we call a feature.

AtoML contains functionality to create fingerprints from ase atoms objects.
This functionality is done by one or several of the fingerprint generators in
AtoML. 

# Adsorbate fingerprinting

In this tutorial we will try the adsorbate fingerprint generator, which
is useful for converting adsorbates on extended surfaces into fingerprints for
predicting their chemisorption energies. Try looking at the code and running
`adsorbate_data_setup.py`.

Attached to the atoms objects, the fingerprinter needs one crucial piece of
information - the indices of atoms belonging to the adsorbate. Those are set in
`atoms.info['ads_index']`. A few other necessary attachments are made by some
auxiliary function based on the `ads_index`:

  ```python
    atoms.info['surf_atoms'] = slab_index(atoms)
    i_add1, i_surf1, Z_add1, Z_surf1, i_surfnn = info2primary_index(atoms)
    atoms.info['i_add1'] = i_add1
    atoms.info['i_surf1'] = i_surf1
    atoms.info['Z_add1'] = Z_add1
    atoms.info['Z_surf1'] = Z_surf1
    atoms.info['i_surfnn'] = i_surfnn
    structures.append(atoms)
  ```

Here, we stored our list of atoms objects in `structures`.
We select a few fingerprinters and put them in a list:

  ```python
    # Get the fingerprint generator.
    fingerprint_generator = AdsorbateFingerprintGenerator()
    # List of functions to call.
    feature_functions = [fingerprint_generator.primary_surfatom,
                         fingerprint_generator.primary_adds_nn]
  ```

The fingerprinters are run like so:

  ```python
    # Generate the data
    training_data = return_fpv(structures, feature_functions, use_prior=False)
  ```

which creates the N x D matrix. Names of features can be obtained by:

  ```python
    feature_names = get_combined_descriptors(feature_functions)
  ```

Finally, the script will print the feature names and plot the distributions.
