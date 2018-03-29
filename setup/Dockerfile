FROM ubuntu:latest

# Set environment variables.
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Add some descriptive labels.
LABEL Description="This image is used to setup the AtoML environment." Version="0.3.1"

# Install various python versions.
RUN apt-get update && apt-get install -y \
    python2.7 \
    python3 \
    python-pip \
    python3-pip

# Install additional python packages.
COPY requirements.txt .
RUN pip2 install --upgrade pip
RUN apt-get remove -y python-pip
RUN pip2 install --upgrade -r requirements.txt; pip2 install asap3
RUN pip3 install --upgrade pip
RUN apt-get remove -y python3-pip
RUN pip3 install --upgrade -r requirements.txt; pip3 install asap3

RUN apt-get update

# Install java.
RUN apt-get -y install software-properties-common && \
    add-apt-repository -y ppa:webupd8team/java && \
    apt update && \
    echo debconf shared/accepted-oracle-license-v1-1 select true | debconf-set-selections && \
    echo debconf shared/accepted-oracle-license-v1-1 seen true | debconf-set-selections && \
    apt -y install oracle-java8-installer && \
    apt -y install oracle-java8-set-default

# Insatll some additional functionality.
RUN apt-get update && apt-get install -y \
    git \
    vim
