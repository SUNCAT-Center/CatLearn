FROM jenningspc/catlearn:latest

# Set environment variables.
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Add some descriptive labels.
LABEL Description="This image is used to run CatLearn locally." Version="0.4.2"

# Create the root directory.
RUN mkdir CatLearn
COPY . /CatLearn/
ENV HOME=/CatLearn
ENV SHELL=/bin/bash
VOLUME /CatLearn
WORKDIR /CatLearn

# Set the PYTHONPATH.
ENV PYTHONPATH=$PWD/:$PYTHONPATH
