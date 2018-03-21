FROM ubuntu:latest

# Set environment variables.
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Add some descriptive labels.
LABEL Description="This image is used to setup the AtoML environment." Version="0.3.1"

RUN apt-get update

# Install various python versions.
RUN apt-get install -y python2.7 && \
    apt-get install -y python3 && \
    apt-get install -y python-pip && \
    apt-get install -y python3-pip

# Install java.
RUN apt-get -y install software-properties-common && \
    add-apt-repository -y ppa:webupd8team/java && \
    apt update && \
    echo debconf shared/accepted-oracle-license-v1-1 select true | debconf-set-selections && \
    echo debconf shared/accepted-oracle-license-v1-1 seen true | debconf-set-selections && \
    apt -y install oracle-java8-installer && \
    apt -y install oracle-java8-set-default && \
    apt-get -y install git

# Install Magpie.
RUN git clone --recursive https://bitbucket.org/wolverton/magpie
RUN cd magpie; ./gradlew jar

# Install additional python packages.
COPY requirements.txt .
RUN pip2 install -r requirements.txt; pip2 install asap3
RUN pip3 install -r requirements.txt; pip3 install asap3
