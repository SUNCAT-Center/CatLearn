# AtoML Tests

All the tests for the AtoML code are here. When writing new code, please add some tests to ensure functionality doesn't break over time. We look at test coverage when merge requests are opened and will expect that coverage does not decrease due to large portions of new code not being tested.

## Table of Contents

[(Back to top)](#atoml-tests)

-   [Test Suite](#test-suite)
-   [Continuous Integration](#continuous-integration)
-   [Command Line](#command-line)

## Continuous Integration

[(Back to top)](#atoml-tests)

Continuous Integration (CI) is used meaning tests are run whenever a new commit is pushed to the origin. For our purposes, we use [GitLab CI](https://docs.gitlab.com/ce/ci/) which checks whether tests run successfully and the coverage. The tests for the AtoML code are imported and run in `test_suite.py`. This can be extended with any new tests that may be written to cover new functionality.

_Please be mindful of the runtime for new tests._

### Test Suite

[(Back to top)](#atoml-tests)

All tests are run in the `test_suite.py` script, using the unittest framework. There are two ways to add new tests.

-   If new functionality is being added to existing functions, simply append a new testing function to the appropriate `TestCase` class. It is important to remember that the function won't be viewed as a test unless it is defined as `test_something(self)`.

-   If the new code goes beyond extending pre-existing functionality, it may be necessary to create a new test class. Please look at the current tests to get an idea of how to set this up. When this is ready, it will be necessary to import the class in `test_suite.py` and add it to the `test_classes_to_run` list.

## Command Line

[(Back to top)](#atoml-tests)

The tests can be run on the command line. When running on the command line, it is important to run the `test_feature_generation.py` script first. This produces a database of features and targets used in most of the other tests. This data can be imported in tests using the `common.get_data()` function.
