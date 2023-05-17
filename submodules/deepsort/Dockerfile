FROM tensorflow/tensorflow:2.3.0-gpu

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /app

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt update && apt install -y git wget libsm6 libxrender1 libxext6
RUN apt install -y sudo build_essential ffmpeg gdebi libopencv-dev cmake

RUN git clone https://github.com/theAIGuysCode/yolov4-deepsort
COPY requirements.txt .
RUN pip install -r requirements.txt 

WORKDIR /app/yolov4-deepsort
RUN wget -P ./data https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
COPY object_tracker.py .
RUN python save_model.py --model yolov4

WORKDIR /app/openpose
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose .
RUN bash ./scripts/ubuntu/install_deps.sh

WORKDIR /app/openpose/build
RUN cmake .. && \
    make -j$(nproc) && \
    make install

RUN set -xe && \
    apt-get clean && \
    rm -rf /tmp/cmake.tar.gz

WORKDIR /app