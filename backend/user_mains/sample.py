import glob
import os

from dataclasses import asdict
from combiner import framefactory
import db_setup
from api.serializers import *
from tqdm import tqdm

base_dir = "/outputs/video"
jsons = glob.glob(os.path.join(base_dir, "keypoints", "*.json"))
csvs = glob.glob(os.path.join(base_dir, "ID", "*.csv"))
jpgs = glob.glob(os.path.join(base_dir, "ID", "*.jpg"))

factory = framefactory.CombinedFrameFactory("demo")
data = []
for json, csv, jpg in tqdm(zip(jsons, csvs, jpgs)):
    json_i = framefactory.OpenPoseJsonData(json)
    csv_i = framefactory.DeepSortCsvData(csv)
    jpg_i = framefactory.DeepSortJpgData(jpg)
    frame = factory.create(json_i, csv_i, jpg_i)
    data.append(asdict(frame))

serializer = FrameListSerializer(data=data)
serializer.is_valid()
serializer.save()
