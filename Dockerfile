FROM python:3.9
# FROM nvidia/cuda:11.7.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /app
ENV PYTHONPATH /app

COPY requirements.txt .
RUN apt update && \ 
    apt install -y tk libglib2.0-0 libsm6 libxrender1 libxext6 fonts-noto-cjk libgl1-mesa-dev less
    
RUN pip install -r requirements.txt