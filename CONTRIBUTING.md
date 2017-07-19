# Contributing

*   Code should support Python 2.7, 3.4 and higher.

*   Code should adhere to the [pep8](https://www.python.org/dev/peps/pep-0008/)
    and [pyflakes](https://pypi.python.org/pypi/pyflakes) style guides.

*   When new functions are added, tests should be written and added to the CI
    script.

*   Should use NumPy style docstrings.

# Git Setup

*   Fork the repository and then clone it to your local machine.

        $ git clone git@gitlab.com:your-user-name/AtoML.git

*   Add and track upstream to the local copy.

        $ git remote add upstream git@gitlab.com:atoML/AtoML.git

# Development

*   Before starting any new work, always sync with the upstream version.

        $ git fetch upstream
        $ git checkout master
        $ git merge upstream/master --ff-only

*   It is a good idea to keep the remote repository up to date.

        $ git push origin master

*   Start a new branch to do work on.

        $ git checkout -b branch-name

*   Once a file has been changed/created, add it to the staging area.

        $ git add file-name

*   Now commit it to the local repository and push it to the remote.

        $ git commit -m 'some descriptive message'
        $ git push --set-upstream origin branch-name

*   When the desired changes have been made on your fork of the repository,
    open up a merge request on GitLab.
