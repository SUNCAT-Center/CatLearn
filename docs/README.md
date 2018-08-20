# CatLearn Docs

**This file is not meant to be included in the docs.** Instead it is meant as a guide for how to update and compile the docs.

To start with it is necessary to install some packages that will be used throughout.

```shell
$ pip install sphinx sphinx-autobuild  # install basic sphinx packages
$ pip install sphinxcontrib-napoleon  # this allows parsing of numpy style docstrings
$ pip install sphinx_rtd_theme  # this adds the theme package
$ pip install recommonmark  # this allows us to use rst and md files
```

If this is a new build, then make the `docs/` folder in CatLearn root and run the quickstart. If the docs folder already exists, these two commands can be skipped.

```shell
$ mkdir docs
$ sphinx-quickstart docs
```

When the docs folder is setup, there will be a `conf.py` and an `index.rst` file. These provide the basis for the sphinx documentation. To change the way the documentation will look and behave, it is necessary to change the `conf.py` file. To add new pages to the docs, files need to be called in the `toctree` in `index.rst`. When the basic style has been defined, the docs for the code can be generated using the following:

```shell
$ sphinx-apidoc -o docs catlearn
```

This should generate individual `.rst` files for all of the CatLearn modules. It may be necessary to edit some of these to clean things up a bit. But following generation of these files, the docs can be built with:

```shell
$ cd docs/
$ make html
```

We use Read the Docs for hosting the CatLearn documentation, found at [catlearn.readthedocs.io](http://catlearn.readthedocs.io/en/latest/). This should be recompiled with each commit to Github, there is a badge on the README that will indicate if the docs builds are passing or failing.
