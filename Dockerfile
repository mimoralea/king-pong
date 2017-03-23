############################################
# Deep Reinforcement Learning dependencies #
############################################

FROM fedora:latest
MAINTAINER Miguel Morales <mimoralea@gmail.com>
RUN dnf upgrade -y && dnf install -y geos-devel opencv-python pygame numpy
RUN pip install --upgrade pip && pip install Shapely
RUN pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl
