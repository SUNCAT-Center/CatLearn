# AtoML Source Code

AtoML is a code base for performing Gaussian Process machine learning on
atomic systems. The code is modular in nature and each module has its own
README, that will provide a more detailed description of what it does.

In general, there are modules for:

*   Manipulate list of atoms objects to form training and test data. Useful
when getting data from e.g. a database.
    -   data_setup.py
*   Convert ASE atoms objects into feature vectors for a number of potentially
interesting problems.
    -   fingerprint_setup.py
    -   adsorbate_fingerprint.py
    -   particle_fingerprint.py
    -   neighborhood_matrix.py
    -   standard_fingerprint.py
    -   general_fingerprint.py
*   Database functions to store the feature matrix from a given dataset.
    -   database_functions.py
*   Feature preprocessing, engineering, elimination and extraction methods.
    -   feature_preprocess.py
    -   feature_engineering.py
    -   feature_elimination.py
    -   feature_extraction.py
*   Gaussian processes predictions with with model optimization.
    -   predict.py
    -   kernels.py
    -   covarience.py
    -   model_selection.py
    -   uncertainty.py
    -   gp_sensitivity.py
    -   cost_function.py
*   Model testing functions.
    -   cross_validation.py
