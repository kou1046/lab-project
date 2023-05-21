FROM python:3.9-slim

WORKDIR /app
ENV PYTHONPATH /app

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt update && \ 
    apt install -y tk libglib2.0-0 libsm6 libxrender1 libxext6 fonts-noto-cjk
