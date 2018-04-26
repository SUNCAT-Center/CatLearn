# Contributing

## General

There are some general coding conventions that the CatLearn repository adheres to. These include the following:

-   Code should support Python 2.7, 3.4 and higher.

-   Code should adhere to the [pep8](https://www.python.org/dev/peps/pep-0008/) and [pyflakes](https://pypi.python.org/pypi/pyflakes) style guides.

-   When new functions are added, tests should be written and added to the CI script.

-   Should use NumPy style [docstrings](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt).

## Git Setup

We adhere to the git workflow described [here](http://nvie.com/posts/a-successful-git-branching-model/), if you are considering contributing, please familiarize yourself with this. It is a bad idea to develop directly on the on the main CatLearn repository. Instead, fork a version into your own namespace on Github with the following:

-   Fork the repository and then clone it to your local machine.

    ```shell
    $ git clone https://github.com/SUNCAT-Center/CatLearn.git
    ```

-   Add and track upstream to the local copy.

    ```shell
    $ git remote add upstream https://github.com/SUNCAT-Center/CatLearn.git
    ```

All development can then be performed on the fork and a merge request opened into the upstream when appropriate. It is normally best to open merge requests as soon as possible, as it will allow everyone to see what is being worked on and comment on any potential issues.

## Development

The following workflow is recommended when adding some new functionality:

-   Before starting any new work, always sync with the upstream version.

    ```shell
    $ git fetch upstream
    $ git checkout master
    $ git merge upstream/master --ff-only
    ```

-   It is a good idea to keep the remote repository up to date.

    ```shell
    $ git push origin master
    ```

-   Start a new branch to do work on.

    ```shell
    $ git checkout -b branch-name
    ```

-   Once a file has been changed/created, add it to the staging area.

    ```shell
    $ git add file-name
    ```

-   Now commit it to the local repository and push it to the remote.

    ```shell
    $ git commit -m 'some descriptive message'
    $ git push --set-upstream origin branch-name
    ```

-   When the desired changes have been made on your fork of the repository, open up a merge request on Github.

## Docker

A [docker](https://www.docker.com) image is included in the repository. It is sometimes easier to develop within a controlled environment such as this. In particular, it is possible for other developers to attain the same environment. To run CatLearn in the docker container, use the following commands:

```shell
$ docker build -t catlearn .
$ docker run -it catlearn bash
```

This will load up the CatLearn directory. To check that everything is working correctly simply run the following:

```shell
$ python2 test/test_suite.py
$ python3 test/test_suite.py
```

This will run the `test_suite.py` script with python version 2 and 3, respectively. If one version of python is preferred over the other, it is possible to create an alias as normal with:

```shell
$ alias python=python3
```

**Use ctrl+d to exit.**

To make changes to this, it is possible to simply edit the `Dockerfile`. To list the images available on the local system, use the following:

```shell
$ docker images
$ docker inspect REPOSITORY
```

It is a good idea to remove old images. This can be performed using the following lines:

```shell
$ docker rm $(docker ps -q -f status=exited)
$ docker rmi $(docker images -q -f "dangling=true")
```

## Testing

When writing new code, please add some tests to ensure functionality doesn't break over time. We look at test coverage when merge requests are opened and will expect that coverage does not decrease due to large portions of new code not being tested. In CatLearn we just use the built-in unittest framework.

When commits are made, the CI will also automatically test if dependencies are up to date. This test is allowed to fail and will simply return a warning if a module in `requirements.txt` is out of date. This shouldn't be of concern and is mostly in place for us to keep track of changes in other code bases that could cause problems.

If changes are being made that change some core functionality, please run the `tutorials/test_notebooks.py` script. In general, the tutorials involve more demanding computations and thus are not run with the CI. The `test_notebooks.py` script will run through the various tutorials and make sure that they do not fail.

## Tutorials

Where appropriate please consider adding some tutorials for new functionality. It would be great if they were written in jupyter notebook form, allowing for some detailed discussion of what is going on in the code.
