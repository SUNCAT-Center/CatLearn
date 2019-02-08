## Fingerprint Generators

A fingerprint generator is a class that contain a number of fingerprinters,
 each of which return a fingerprint vector, representing an atomic struture.
 
## Built-in Vector Generators

[(Back to top)](#table-of-contents)

-   [composition_vec](#composition_vec)
-   [element_parameter_vec](#element_parameter_vec)
-   [element_mass_vec](#element_mass_vec)
-   [eigenspectrum_vec](#eigenspectrum_vec)
-   [distance_vec](#distance_vec)
-   nearestneighbour_vec
-   bond_count_vec
-   distribution_vec
-   connections_vec
-   rdf_vec

Below is a more detailed description of the vectors that will be generated.

### composition_vec

[(Back to list)](#available-vector-generators)

-   Generator to return a feature vector based on the composition. The resulting vector contains a count of the different atomic types, e.g. for CH3OH the vector `[1, 4, 1]` would be returned.
-   Feature names will be returned in the case of CH3OH the name vector will be `['1_count', '6_count', '8_count']`.
-   The data object passed to this generator must contain the atomic numbers.

### element_parameter_vec

[(Back to list)](#available-vector-generators)

-   Generator to return a feature vector based on a combination of the composition and a set of user-defined parameters. The vector is compiled based on the summed parameters for each elemental type as well as the sum for all atoms. For CH3OH the Pauling electronegativity parameters are `{1: 2.2, 6: 2.55, 8: 3.44}` so when combined with the composition vector, `[1, 4, 1]`, this results in the vector `[2.2, 10.2, 3.44, 15.84]`.
-   Feature names will be returned in the case of CH3OH the name vector will be `['sum_1_en_pauling', 'sum_6_en_pauling', 'sum_8_en_pauling', 'sum_all_en_pauling']`.
-   The data object passed to this generator must contain the atomic numbers.

### element_mass_vec

[(Back to list)](#available-vector-generators)

-   Generator to return a feature vector based on the summed mass of the atoms data.
-   Feature names will be returned as `['sum_mass']` always.
-   The data object passed to this generator must contain the atomic masses.

### eigenspectrum_vec

[(Back to list)](#available-vector-generators)

-   Generator to return a feature vector based on the sorted eigendecomposition of the Coulomb matrix. A more detailed description of the feature vector can be found [here](https://doi.org/10.1103/PhysRevLett.108.058301). The vector scales with the number of atoms.
-   Feature names will be returned as `['eig_0', eig_1, ..., eig_N]`.
-   The data object passed to this generator must contain the atomic coordinates and atomic numbers.

### distance_vec

[(Back to list)](#available-vector-generators)

-   Generator to return a feature vector based on the average distance between homoatomic species.
-   Feature names will be returned as `['1-1_dist', 6-6_dist, 8-8_dist]` for CH3OH.
-   The data object passed to this generator must contain the atomic coordinates and atomic numbers.
