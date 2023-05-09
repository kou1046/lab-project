FROM tensorflow/tensorflow:2.5.0-gpu

LABEL tag="submodule_image"

WORKDIR /app

COPY requirements.txt .
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt update && apt install -y libsm6 libxrender1 libxext6
RUN pip install -r requirements.txt
COPY openpose/ ./openpose
COPY yolov4-deepsort/ ./yolov4-deepsort

WORKDIR /app/yolov4-deepsort
RUN python save_model.py --model yolov4
CMD ["python", "object_tracker.py", "--video", "outputs/demo.avi", "--dont_show", "--info"]