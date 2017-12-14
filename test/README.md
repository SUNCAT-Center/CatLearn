# AtoML Tests

All the tests for the AtoML code are here. When writing new code, please add
some tests to ensure functionality doesn't break over time.

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
