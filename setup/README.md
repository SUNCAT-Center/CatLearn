# AtoML Setup

In this folder, there are a number of scripts used in the setup of the AtoML environment.

## Docker

There is a custom docker image used for testing that contains an environment with Java and Python installed. The python requirements have been installed for both versions. To update the docker image on docker hub, the following commands must be run:

```shell
  $ docker build -t atoml:latest .
  $ docker tag atoml:latest jenningspc/atoml:latest
  $ docker push jenningspc/atoml:latest
```

To use the image, simply do:

```
  FROM jenningspc/atoml:latest
```

### Limitations

-   To build the docker image an up to date version of the requirements is needed.
-   Versioning not really handled.
