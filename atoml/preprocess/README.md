# Data preprocessing

The folder contains files to scale and optimize the feature space.

## Preprocessing functions

[(Back to top)](#data-preprocessing)

-   [Feature preprocess](#feature-preprocess)
-   [Feature engineering](#feature-engineering)
-   [Feature elimination](#feature-elimination)
-   [Feature extraction](#feature-extraction)
-   [Importance testing](#importance-testing)

## Feature preprocess

[(Back to top)](#data-preprocessing)

This class provide functions to scale the feature data in different ways:

-   standardize
-   normalize
-   min/max
-   unit length

These have different advantages/disadvantages with respect to the data type and underlying relationship. It is particularly important to not if global scaling is necessary, where local scaling will can result in poor models, depending on how data is generated and when training is performed.

### ToDo

-   Better define when scaling can be used with local-only/global data.

## Feature engineering

[(Back to top)](#data-preprocessing)

Feature engineering can be used to come up with more effective representations which can allow for more model flexibility while simplifying the representation. There are a number of transformations available, however if these are used, it is likely that a feature elimination method will also be needed.

This can add computational cost to generating the model and currently there doesn't appear to be a compelling reason to use it.

### ToDo

-   More robust investigation for application to different problems.

## Feature elimination

[(Back to top)](#data-preprocessing)

The aim of feature elimination is to reduce the feature space, removing features that either add noise to the predictions, or that don't add additional information to the model. The method used for elimination largely depends on the size of the initial feature space. When the number of features is larger than the number of training data points, a screening method can be used, such as Sure Independence Screening (SIS). This uses the correlation between features and targets as a measure of fitness and removes features that don't correlate well. This is a simplistic elimination method but can work well when the feature space is too large to consider other (more accurate) elimination methods.

Screening can be performed once, or in an iterative manner. When performed once, it is likely that more correlated features will be returned. When performed in an iterative manner, correlation between accepted features and the remaining feature set are also removed. This results in an uncorrelated feature set being likely to be returned. It is possible to test feature acceptance against the likelihood of just including random noise. The selection will stop early if random features correlate better with the targets than real features.

If the feature space is smaller, it is possible to generate a sparse model using e.g. LASSO. Features with zero coefficients in the sparse linear model can then be considered to be unnecessary in the GP representation and eliminated. Similar procedures can be used with other linear regressors, such as ridge, where it is possible to take the n-features that contribute most to the linear model.

More costly feature elimination methods, such as sensitivity analysis, exist within the GP code.

### ToDo

-   Speed up some of the elimination methods.

# Feature extraction

[(Back to top)](#data-preprocessing)

When a reasonable feature space has been generated it is potentially useful to perform feature extraction to produce a reduced number of more informative features. This is typically achieved by taking combinations of original features and so can result in less interpretable models. Common methods of extraction include PCA and PLS. A wrapper to scikit-learn has been written for a few of these methods, as well as a fully coded (very) basic PCA function.

# Importance testing

[(Back to top)](#data-preprocessing)

Some functions have been defined for testing the importance of features is a simple manner. These functions simply take a feature at a given index and permute the feature in some way to reduce its significance. When a model is trained with the new "useless" feature, the error can be tracked and features that correspond to an increase in error can be viewed as being more significant than those that have no impact on the accuracy.

The the functions are as follows:

-   Make a feature invariant.
-   Make a feature random noise.
-   Shuffle a feature.
