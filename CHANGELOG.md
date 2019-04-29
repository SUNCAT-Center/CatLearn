# dev

# Version 0.6.1 (April 2019)

-   Fixed compatibility issue with MLNEB and [GPAW](https://wiki.fysik.dtu.dk/gpaw/index.html)
-   Various bugfixes

# Version 0.6.0 (January 2019)

-   Added ML-MIN algorithm for energy minimization.
-   Added ML-NEB algorithm for transition state search.
-   Changed input format for kernels in the GP.

# Version 0.5.0 (October 2018)

-   Restructure of fingerprint module
-   Pandas DataFrame getter in FeatureGenerator
-   CatMAP API using ASE database.
-   New active learning module.
-   Small fixes in adsorbate fingerprinter.

# Version 0.4.4 (August 2018)

-   Major modifications to adsorbates fingerprinter
-   Bag of site neighbor coordinations numbers implemented.
-   Bag of connections implemented for adsorbate systems.
-   General bag of connections implemented.
-   Data cleaning function now return a dictionary with 'index' of clean features.
-   New clean function to discard features with excessive skewness.
-   New adsorbate-chalcogenide fingerprint generator.
-   Enhancements to automatic identification of adsorbate, site.
-   Generalized coordination number for site.
-   Formal charges utility.
-   New sum electronegativity over bonds fingerprinter.

# Version 0.4.3 (May 2018)

-   `ConvolutedFingerprintGenerator` added for bulk and molecules.
-   Dropped support for Python3.4 as it appears to start causing problems.

# Version 0.4.2 (May 2018)

-   Genetic algorithm feature selection can parallelize over population within each generation.
-   Default fingerprinter function sets accessible using `catlearn.fingerprint.setup.default_fingerprinters`
-   New surrogate model utility
-   New utility for evaluating cutoff radii for connectivity based fingerprinting.
-   `default_catlearn_radius` improved.

# Version 0.4.1 (April 2018)

-   AtoML renamed to CatLearn and moved to Github.
-   Adsorbate fingerprinting again parallelizable.
-   Adsorbate fingerprinting use atoms.tags to get layers if present.
-   Adsorbate fingerprinting relies on connectivity matrix before neighborlist.
-   New bond-electronegativity centered fingerprints for adsorbates.
-   Fixed a bug that caused the negative log marginal likelihood to be attached to the gp class.
-   Small speed improvement for initialize and updates to `GaussianProcess`.

# Version 0.4.0 (April 2018)

-   Added `autogen_info` function for list of atoms objects representing adsorbates.
    -   This can auto-generate all atomic group information and attach it to `atoms.info`.
    -   Parallelized fingerprinting is not yet supported for output from `autogen_info`.
-   Added `database_to_list` for import of atoms objects from ase.db with formatted metadata.
-   Added function to translate a connection matrix to a formatted neighborlist dict.
-   `periodic_table_data.list_mendeleev_params` now returns a numpy array.
-   Magpie api added, allows for Voronoi and prototype feature generation.
-   A genetic algorithm added for feature optimization.
-   Parallelism updated to be compatable with Python2.
-   Added in better neighborlist generation.
    -   Updated wrapper for ase neighborlist.
    -   Updated CatLearn neighborlist generator.
    -   Defaults cutoffs changed to `atomic_radius` plus a relative tolerance.
-   Added basic NetworkX api.
-   Added some general functions to clean data and build a GP.
-   Added a test for dependencies. Will raise a warning in the CI if things get out of date.
-   Added a custom docker image for the tests. This is compiled in the `setup/` directory in root.
-   Modified uncertainty output. The user can ask for the uncertainty with and without adding noise parameter (regularization).
-   Clean up some bits of code, fix some bugs.

# Version 0.3.1 (February 2018)

-   Added a parallel version of the greedy feature selection. **Python3 only!**
-   Updated the k-fold cross-validation function to handle features and targets explicitly.
-   Added some basic read/write functionality to the k-fold CV.
-   A number of minor bugs have been fixed.

# Version 0.3.0 (February 2018)

-   Update the fingerprint generator functions so there is now a `FeatureGenerator` class that wraps round all type specific generators.
-   Feature generation can now be performed in parallel, setting `nprocs` variable in the `FeatureGenerator` class. **Python3 only!**
-   Add better handling when passing variable length/composition data objects to the feature generators.
-   More acquisition functions added.
-   Penalty functions added.
-   Started adding a general api for ASE.
-   Added some more test and changed the way test are called/handled.
-   A number of minor bugs have been fixed.

# Version 0.2.1 (February 2018)

-   Update functions to compile features allowing for variable length of atoms objects.
-   Added some tutorials for hierarchy cross-validation and prediction on organic molecules.

# Version 0.2.0 (January 2018)

-   Gradients added to hyperparameter optimization.
-   More features added to the adsorbate fingerprint generator.
-   Acquisition function structure updated. Added new functions.
-   Add some standardized input/output functions to save and load models.
-   The kernel setup has been made more modular.
-   Better test coverage, the tests have also been optimized for speed.
-   Better CI configuration. The new method is much faster and more flexible.
-   Added Dockerfile and appropriate documentation in the README and CONTRIBUTING guidelines.
-   A number of minor bugs have been fixed.

# Version 0.1.0 (December 2017)

-   The first stable version of the code base!
-   For those that used the precious development version, there are many big changes in the way the code is structured. Most scripts will need to be rewritten.
-   A number of minor bugs have been fixed.
