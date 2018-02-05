# Fingerprint setup

The folder contains functions for generating a feature space.

## Table of contents

-   [Usage](#usage)
-   [Available Vector Generators](#available-vector-generators)

## Usage
[(Back to top)](#table-of-contents)

There are several ways to generate the feature vectors. The easiest way is to
use the `FeatureGenerator`. Initially one must initialize the class:

  ```python
    generator = FeatureGenerator()
  ```

The feature generation is handled by some setup functions. Once the class has
been initialized, it is then possible to call any of the generator functions.

  ```python
    features = generator.return_vec(atoms, generator.eigenspectrum_vec)
  ```

Instead of passing a single function, it is also possible to pass a list of
generators:

  ```python
    features = generator.return_vec(atoms, [generator.bond_count_vec,
                                            generator.eigenspectrum_vec,
                                            generator.composition_vec])
  ```

It is even possible to pass user defined functions in as well. These should
just take the atoms objects being passed to `generator.return_vec` and return a
numpy array vector.

  ```python
    def f(atoms):
        return [1., len(atoms), random.random()]

    features = generator.return_vec(atoms, f)
  ```

The various generators will also return a name vector if the `return_names`
function is called.

```python
  names = generator.return_names([generator.bond_count_vec,
                                  generator.eigenspectrum_vec,
                                  generator.composition_vec])
```

## Available Vector Generators
[(Back to top)](#table-of-contents)

-   [composition_vec](#composition_vec)
-   [element_parameter_vec](#element_parameter_vec)
-   element_mass_vec
-   eigenspectrum_vec
-   distance_vec
-   nearestneighbour_vec
-   bond_count_vec
-   distribution_vec
-   connections_vec
-   rdf_vec

Below is a more detailed description of the vectors that will be generated.

#### composition_vec
[(Back to list)](#available-vector-generators)

*   Generator to return a feature vector based on the composition. The
resulting vector contains a count of the different atomic types, e.g. for CH3OH
the vector `[1, 4, 1]` would be returned.
*   Feature names will be returned in the case of CH3OH the name vector will be
`['1_count', '6_count', '8_count']`.
*   The data object passed to this generator must contain the atomic numbers.

#### element_parameter_vec
[(Back to list)](#available-vector-generators)

*   Generator to return a feature vector based on a combination of the
composition and a set of user-defined parameters. The vector is compiled based
on the summed parameters for each elemental type as well as the sum for all
atoms. For CH3OH the Pauling electronegativity parameters are `{1: 2.2, 6: 2.55,
8: 3.44}` so when combined with the composition vector, `[1, 4, 1]`, this
results in the vector `[2.2, 10.2, 3.44, 15.84]`.
*   Feature names will be returned in the case of CH3OH the name vector will be
`['sum_1_en_pauling', 'sum_6_en_pauling', 'sum_8_en_pauling',
'sum_all_en_pauling']`.
*   The data object passed to this generator must contain the atomic numbers.
