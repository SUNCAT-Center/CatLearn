# Cross-validation functions

The folder contains functions to perform cross validation. Cross-validation
should be used in testing and optimizing the models. It will provide confidence
limits on how well training data will generalize to unseen test data. If
hyperparameter optimization is performed, cross-validation also provides a way
of avoiding overfitting.

## Hierarchy CV
[(Back to top)](#cross-validation-functions)

This forms the base class for cross-validation performed to generate the
learning curve. The learning curve will plot the relationship between the
average squared error on cross validation set against the training data size.
To achieve this, hierarchy-cv will recursively split a dataset into smaller
subsets to a user defined limit.

In the current form, the function requires that all data is stored in a basic
database structure and will simply generate a dictionary containing a subset
label and list of database indexes from which to draw the data. This is done
to avoid saving large numbers of feature matrices in memory.

#### ToDo

*   Requires some generalization, especially with respect to the database.

## K-fold CV
[(Back to top)](#cross-validation-functions)

K-fold cross-validation will divide a dataset into k equal sized subsets.
The current method is over simplistic and does not include replacement or a
prediction function.

#### ToDo

*   Add in predictor function.
*   Add in replacement.
