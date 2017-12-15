# AtoML Tests

All the tests for the AtoML code are here. When writing new code, please add
some tests to ensure functionality doesn't break over time.

## Test Scripts
[(Back to top)](#atoml-tests)

-   [`test_data_clean.py`](#data-cleaning)
-   [`test_data_setup.py`](#data-setup)
-   [`test_feature_optimization.py`](#feature-optimization)
-   [`test_predict.py`](#predictions)
-   [`test_suit.py`](#test-suit)

### Data Cleaning
[(Back to top)](#atoml-tests)

Functions used to clean the data are tested in `test_data_clean.py` with some
to data. In the current tests, fake data is used so we can ensure messy data
outliers. The tests are mostly just checking that we remove select feature or
data points from the original dataset.

### Data Setup
[(Back to top)](#atoml-tests)

Functions used to generate features (data), are tested in `test_data_clean.py`
with randomly selected atoms objects. This is currently importing nanoparticle
data, if different data is required to test out a feature generator, this will
need to be imported separately. The tests are mostly testing that features are
generated with the correct dimensionality.

*This script should be called before any of the following can be run.*

### Feature Optimization
[(Back to top)](#atoml-tests)

Many of the preprocessing feature optimization functions are tested in the
`test_feature_optimization.py` script. The tests are mostly checking that the
dimensions of the feature set change in the correct manner, whether elimination
or extraction functions are called.

### Predictions
[(Back to top)](#atoml-tests)

The regression functions are tested in the `test_predict(_scale).py` scripts. This
mostly just checks that predictions are made for all data points. Further, the
predictions are printed in the CI log so it is possible to see if they look
reasonable.

### Test Suit
[(Back to top)](#atoml-tests)

All tests are run in the `test_suit.py` script. This just iterates through all
tests in the correct order, using the unittest framework. This should be
updated when any new tests are added.

## Continuous Integration
[(Back to top)](#atoml-tests)

Continuous Integration (CI) is used so test are run whenever a new commit is
pushed to the origin. For our purposes, we use
[GitLab CI](https://docs.gitlab.com/ce/ci/) which checks whether tests run and
the coverage of the tests. The tests for the AtoML code are imported and run in
`test_suit.py`. This can be extended with any new tests that may be written to
cover new functionality.

## Command Line
[(Back to top)](#atoml-tests)

They can also be run on the command line. If this is done, please make sure
that `pyinstrument` profiler is installed:

```shell
  pip install --upgrade pyinstrument
```

The profiler is run to give an indication of how long tests are likely to take
and what impact changes are likely to have on the overall test suite. If a
function is likely to be expensive, please consider ways in which it may be
optimized in the test, e.g. passing slightly less data.

When running on the command line, it is important to run the
`test_data_setup.py` script first. This generates data used in most of the
other tests.
