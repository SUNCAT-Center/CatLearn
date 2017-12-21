FROM continuumio/anaconda

# Set environment variables.
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN mkdir AtoML
COPY . /AtoML/
ENV HOME=/AtoML
ENV SHELL=/bin/bash
VOLUME /AtoML
WORKDIR /AtoML

ENV PYTHONPATH=$PWD/:$PYTHONPATH

RUN pip install ase pytest-cov pyinstrument
