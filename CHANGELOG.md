# Version 0.3.1 (February 2018)

-   Added a parallel version of the greedy feature selection. **Python3 only!**
-   Updated the k-fold cross-validation function to handle features and targets explicitly.
-   Added some basic read/write functionality to the k-fold CV.

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
