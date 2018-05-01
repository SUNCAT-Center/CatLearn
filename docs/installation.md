# Installation

A number of different methods can be used to run the CatLearn code.

## Requirements

-   ase
-   h5py
-   networkx
-   numpy
-   pandas
-   scikit-learn
-   scipy
-   tqdm

## Installation using pip

The easiest way to install CatLearn is with:

```shell
$ pip install catlearn
```

This will automatically install the code as well as the dependencies.

## Installation from source

To get the most up-to-date development version of the code, you can clone the git repository to a local directory with:

```shell
$ git clone https://github.com/SUNCAT-Center/CatLearn.git
```

And then put the `<install_dir>/` into your `$PYTHONPATH` environment variable. If you are using Windows, there is some advice on how to do that [here](https://stackoverflow.com/questions/3701646/how-to-add-to-the-pythonpath-in-windows-7).

Be sure to install dependencies in with:

```shell
$ pip install -r requirements.txt
```
