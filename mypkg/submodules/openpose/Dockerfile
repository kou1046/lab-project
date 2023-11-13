FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES video,compute,utility
ENV CMAKE_VER="3.13.4"

RUN apt-get update && \
    apt-get install -y \
            git sudo wget \
            build-essential \
            ffmpeg \
            gdebi \
            libopencv-dev

WORKDIR /tmp
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VER}/cmake-${CMAKE_VER}.tar.gz
RUN tar -zxvf cmake-${CMAKE_VER}.tar.gz
WORKDIR /tmp/cmake-${CMAKE_VER}
RUN ./bootstrap
RUN make && make install && rm -rf /tmp/cmake-${CMAKE_VER}.tar.gz

WORKDIR /app/openpose
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose .
RUN bash ./scripts/ubuntu/install_deps.sh

WORKDIR /app/openpose/build
RUN cmake .. -DDOWNLOAD_HAND_MODEL=OFF -DDOWNLOAD_FACE_MODEL=OFF

## FIXME as v1.7.0 support AMPERE architectur, however we are using cuda 10 which is not compatible with AMPERE we comment these lines as a quick fix
RUN sed -ie 's/set(AMPERE "80 86")/#&/g'  ../cmake/Cuda.cmake 
RUN sed -ie 's/set(AMPERE "80 86")/#&/g'  ../3rdparty/caffe/cmake/Cuda.cmake 
## https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1753

RUN make -j$(nproc) && \
    make install

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app/openpose
