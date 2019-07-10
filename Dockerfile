FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

MAINTAINER Mike Ivanov, mike.ivanou@gmail.com

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        cmake \
        build-essential \
        python3-dev \
        wget \
        git \
        libgtk2.0-dev \
	&& rm -rf /var/lib/apt/lists/*

RUN pip3 --no-cache-dir install \
    setuptools

RUN pip3 --no-cache-dir install \
    dlib \
    indicoio \
    imageio \
    textblob \
    Flask \
    wtforms \
    tensorflow \
    tf-nightly \
    torch \
    torchvision \
    pandas \
    opencv-python \
    scipy \
    imutils \
    matplotlib

RUN pip3 --no-cache-dir install \
    python-telegram-bot --upgrade

RUN mkdir -p /root/.torch/models/ && \
    wget -P /root/.torch/models/ "https://download.pytorch.org/models/vgg16-397923af.pth"

RUN git clone https://github.com/vladostan/photohack

WORKDIR /photohack

RUN mkdir -p pics


CMD ["/bin/bash"]