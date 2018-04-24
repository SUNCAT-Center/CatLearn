# CatLearn Setup

In this folder, there are a number of scripts used in the setup of the CatLearn environment.

## Contributing

When contributing to the project, there are a couple of useful scripts here. The main one is the `pre-commit` script, which checks for style compliance when committing new code. To enable this, please do the following:

```shell
  $ cp pre-commit ../.git/hooks/.
  $ chmod +x ../.git/hooks/pre-commit
```

The `git-clean.py` script will simply check for branches on the local copy of the repository that have been merged with master and delete them.

## Docker

There is a custom docker image used for testing that contains an environment with Java and Python installed. The python requirements have been installed for both versions. To update the docker image on docker hub, the following commands must be run:

```shell
  $ docker build -t catlearn:latest .
  $ docker tag catlearn:latest jenningspc/catlearn:latest
  $ docker push jenningspc/catlearn:latest
```

It is assumed that there is an up-to-date version of the `requirements.txt` file in the directory. Then to use the image, simply do:

```
  FROM jenningspc/catlearn:latest
```

There is a bash script that will run these commands and clean up after:

```shell
  $ ./build_docker.sh
```

### Limitations

-   Versioning not really handled.
