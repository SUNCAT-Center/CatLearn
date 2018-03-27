#!/bin/bash

cp ../requirements.txt .

docker build -t atoml:latest .
docker tag atoml:latest jenningspc/atoml:latest
docker push jenningspc/atoml:latest

rm requirements.txt
