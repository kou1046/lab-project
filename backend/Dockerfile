FROM nvidia/cuda:11.7.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /app
ENV PYTHONPATH /app

COPY requirements.txt .
RUN apt update && \ 
    apt install -y tk libglib2.0-0 libsm6 libxrender1 libxext6 fonts-noto-cjk \
        python3 python3-pip python-tk sudo wget git vim
RUN pip install -r requirements.txt