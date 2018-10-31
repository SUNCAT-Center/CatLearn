Welcome to CatLearn's documentation!
====================================

.. image:: https://travis-ci.org/SUNCAT-Center/CatLearn.svg?branch=master
    :target: https://travis-ci.org/SUNCAT-Center/CatLearn

.. image:: https://coveralls.io/repos/github/SUNCAT-Center/CatLearn/badge.svg?branch=master
    :target: https://coveralls.io/github/SUNCAT-Center/CatLearn?branch=master

.. image:: https://img.shields.io/github/license/SUNCAT-Center/CatLearn.svg
    :target: https://github.com/SUNCAT-Center/CatLearn/blob/master/LICENSE

.. image:: https://badge.fury.io/py/CatLearn.svg
    :target: https://badge.fury.io/py/CatLearn

CatLearn_ provides utilities for building and testing atomistic machine learning models for surface science and catalysis.

.. note:: This is part of the SUNCAT centers code base for understanding materials for catalytic applications. Other code is hosted on the center's Github_ repository.

-------------------

CatLearn provides an environment to facilitate utilization of machine learning within the field of materials science and catalysis. Workflows are typically expected to utilize the Atomic Simulation Environment (ASE_), or NetworkX_ graphs.
Through close coupling with these codes, CatLearn can generate numerous embeddings for atomic systems. As well as generating a useful feature space for numerous problems, CatLearn has functions for model optimization. Further, Gaussian
processes (GP) regression machine learning routines are implemented with additional functionality over standard implementations such as that in scikit-learn.
A more detailed explanation of how to utilize the code can be found in the Tutorials_ folder.

To featurize ASE atoms objects, the following lines of code can be used::

    import ase
    from ase.cluster.cubic import FaceCenteredCubic

    from catlearn.fingerprint.setup import FeatureGenerator

    # First generate an atoms object.
    surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
    layers = [6, 9, 5]
    lc = 3.61000
    atoms = FaceCenteredCubic('Cu', surfaces, layers, latticeconstant=lc)

    # Then generate some features.
    generator = FeatureGenerator(nprocs=1)
    features = generator.return_vec([atoms], [generator.eigenspectrum_vec,
                                              generator.composition_vec])

In the most basic form, it is possible to set up a GP model and make some predictions using the following lines of code::

    import numpy as np
    from catlearn.regression import GaussianProcess

    # Define some input data.
    train_features = np.arange(200).reshape(50, 4)
    target = np.random.random_sample((50,))
    test_features = np.arange(100).reshape(25, 4)

    # Setup the kernel.
    kernel = {'k1': {'type': 'gaussian', 'width': 0.5}}

    # Train the GP model.
    gp = GaussianProcess(kernel_dict=kernel, regularization=1e-3,
                         train_fp=train_features, train_target=target,
                         optimize_hyperparameters=True)

    # Get the predictions.
    prediction = gp.predict(test_fp=test_features)

There is much functionality in CatLearn to assist in handling atom data and building optimal models. This includes:

*   API to other codes:

    *   Atomic simulation environment API
    *   Magpie API
    *   NetworkX API

*   Fingerprint generators:

    *   Bulk systems
    *   Support/slab systems
    *   Discrete systems

*   Preprocessing routines:

    *   Data cleaning
    *   Feature elimination
    *   Feature engineering
    *   Feature extraction
    *   Feature scaling

*   Regression methods:

    *   Regularized ridge regression
    *   Gaussian processes regression

*   Cross-validation:

    *   K-fold cv
    *   Ensemble k-fold cv

*   General utilities:

    *   K-means clustering
    *   Neighborlist generators
    *   Penalty functions
    *   SQLite db storage


.. toctree::
   :maxdepth: 1
   :caption: User Guide:

   installation
   changelog
   contributing

.. toctree::
  :maxdepth: 10
  :caption: Code:

  catlearn.api
  catlearn.cross_validation
  catlearn.featurize
  catlearn.fingerprint
  catlearn.ga
  catlearn.learning_curve
  catlearn.preprocess
  catlearn.regression
  catlearn.active_learning
  catlearn.estimator
  catlearn.utilities


.. automodule:: catlearn
    :members:
    :undoc-members:
    :show-inheritance:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`


.. _Catlearn: https://github.com/SUNCAT-Center/CatLearn
.. _Github: https://github.com/SUNCAT-Center
.. _ASE: https://wiki.fysik.dtu.dk/ase/
.. _NetworkX: https://networkx.github.io/
.. _Tutorials: https://github.com/SUNCAT-Center/CatLearn/tree/master/tutorials
