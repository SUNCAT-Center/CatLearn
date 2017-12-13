# AtoML Tests

All the tests for the AtoML code are here, they are imported and run in
`test_suit.py`.

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
