FROM continuumio/anaconda

# Set environment variables.
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Create the root directory.
RUN mkdir AtoML
COPY . /AtoML/
ENV HOME=/AtoML
ENV SHELL=/bin/bash
VOLUME /AtoML
WORKDIR /AtoML

# Set the PYTHONPATH.
ENV PYTHONPATH=$PWD/:$PYTHONPATH

# Install additional python packages.
RUN pip install ase pytest-cov pyinstrument tqdm
