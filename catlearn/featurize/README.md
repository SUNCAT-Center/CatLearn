# Fingerprint setup

The folder contains functions for generating a feature space.

## Table of contents

-   [Usage](#usage)
-   [Available Vector Generators](#available-vector-generators)

## Usage

[(Back to top)](#table-of-contents)

There are several ways to generate the feature vectors. The easiest way is to use the `FeatureGenerator`. Initially one must initialize the class:

```python
    from catlearn.featurize.setup import FeatureGenerator
    generator = FeatureGenerator(nprocs=None)
```

It is possible to generate feature vectors with parallelization when `nprocs` is set. This can either be an int to define the number of cores to use. If `nprocs=None` is set then all available cores are used. The default is `nprocs=1`, running in serial. **This is an experimental feature and only available with Python3.**

The feature generation is handled by some setup functions. Once the class has been initialized, it is then possible to call any of the generator functions.

```python
    features = generator.return_vec(atoms, generator.eigenspectrum_vec)
```

Instead of passing a single function, it is also possible to pass a list of generators:

```python
    features = generator.return_vec(atoms, [generator.bond_count_vec,
                                            generator.eigenspectrum_vec,
                                            generator.composition_vec])
```

It is even possible to pass user defined functions in as well. These should just take the atoms objects being passed to `generator.return_vec` and return a numpy array vector.

```python
    def f(atoms):
        return [1., len(atoms), random.random()]

    features = generator.return_vec(atoms, f)
```

The various generators will also return a name vector if the `return_names` function is called.

```python
    names = generator.return_names([generator.bond_count_vec,
                                    generator.eigenspectrum_vec,
                                    generator.composition_vec])
```