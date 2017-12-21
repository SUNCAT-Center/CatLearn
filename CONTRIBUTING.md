# Contributing

## Table of contents

-   [General](#general)
-   [Git Setup](#git-setup)
-   [Development](#development)
-   [Docker](#docker)

## General
[(Back to top)](#table-of-contents)

There are some general coding conventions that the AtoML repository adheres to.
These include the following:

*   Code should support Python 2.7, 3.4 and higher.

*   Code should adhere to the [pep8](https://www.python.org/dev/peps/pep-0008/)
    and [pyflakes](https://pypi.python.org/pypi/pyflakes) style guides.

*   When new functions are added, tests should be written and added to the CI
    script.

*   Should use NumPy style [docstrings](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt).

## Git Setup
[(Back to top)](#table-of-contents)

It is a bad idea to develop directly on the on the main AtoML repository.
Instead, fork a version into your own namespace on gitlab with the following:

*   Fork the repository and then clone it to your local machine.

      ```shell
        $ git clone git@gitlab.com:your-user-name/AtoML.git
      ```

*   Add and track upstream to the local copy.

      ```shell
        $ git remote add upstream git@gitlab.com:atoML/AtoML.git
      ```

All development can then be performed on the fork and a merge request opened
into the upstream when appropriate. It is normally best to open merge requests
as soon as possible, as it will allow everyone to see what is being worked on
and comment on any potential issues.

## Development
[(Back to top)](#table-of-contents)

The following workflow is recommended when adding some new functionality:

*   Before starting any new work, always sync with the upstream version.

      ```shell
        $ git fetch upstream
        $ git checkout master
        $ git merge upstream/master --ff-only
      ```

*   It is a good idea to keep the remote repository up to date.

      ```shell
        $ git push origin master
      ```

*   Start a new branch to do work on.

      ```shell
        $ git checkout -b branch-name
      ```

*   Once a file has been changed/created, add it to the staging area.

      ```shell
        $ git add file-name
      ```

*   Now commit it to the local repository and push it to the remote.

      ```shell
        $ git commit -m 'some descriptive message'
        $ git push --set-upstream origin branch-name
      ```

*   When the desired changes have been made on your fork of the repository,
    open up a merge request on GitLab.

## Docker
[(Back to top)](#table-of-contents)

A [docker](https://www.docker.com) image is included in the repository. It is
sometimes easier to develop within a controlled environment such as this. In
particular, it is possible for other developers to attain the same environment.
To run AtoML in the docker container, use the following commands:

      ```shell
        $ docker build -t atoml .
        $ docker run -it atoml bash
      ```

This will load up the AtoML directory. To check that everything is working
correctly simply run the following:

      ```shell
        $ python test/test_suit.py
      ```

To make changes to this, it is possibly to simply edit the `Dockerfile`. The
current setup uses Python 2.7, to change this to Python 3.6 simply edit the
first line of the `Dockerfile` to:

      ```shell
        FROM continuumio/anaconda3
      ```
