#!/bin/bash

cp ../requirements.txt .

docker build -t catlearn:latest .
docker tag catlearn:latest jenningspc/catlearn:latest
docker push jenningspc/catlearn:latest

rm requirements.txt
