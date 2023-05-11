FROM tensorflow/tensorflow:2.3.0-gpu

LABEL tag="submodule_image"

WORKDIR /app

COPY requirements.txt .
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN 6 | apt update && apt install -y libsm6 libxrender1 libxext6 \ 
        build-essential cmake libgtk2.0-dev pkg-config \
        libavcodec-dev libavformat-dev libswscale-dev libopencv-dev

COPY openpose/ ./openpose
COPY yolov4-deepsort/ ./yolov4-deepsort
RUN pip install -r requirements.txt 

WORKDIR /app/openpose
RUN scripts/ubuntu/install_deps.sh
RUN mkdir build
WORKDIR /app/openpose/build
RUN cmake .. -DCUDA_ARCH=Manual -DCUDA_ARCH_BIN="52 61" -DCUDA_ARCH_PTX="" \
    make -j$(nproc)

WORKDIR /app/yolov4-deepsort
COPY submodules_edits/object_tracker.py .
RUN python save_model.py --model yolov4