# FROM python:3.7.3-slim
# FROM nvidia/cuda:10.2-cudnn7-devel
# FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER M.N.Bos-29@umcutrecht.nl
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt-get etc.
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

RUN ln -s /usr/bin/python3 /usr/bin/python && \
ln -s /usr/bin/pip3 /usr/bin/pip

## Do not edit if you have a requirements.txt
RUN pip install -r requirements.txt

