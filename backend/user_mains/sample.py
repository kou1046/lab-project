from tkinter.filedialog import askdirectory
import glob
from mypkg.submodules_aggregator import (
    complementidcreator,
    preprocessor,
    framefactory,
    save_frames_to_db,
)
import cv2

base_dir = askdirectory(initialdir="/outputs")
jsons = glob.glob(f"{base_dir}/keypoints/*.json")
ids = glob.glob(f"{base_dir}/ID/*.csv")
jpgs = glob.glob(f"{base_dir}/ID/*.jpg")
cs = glob.glob(f"{base_dir}/*.json")
c, m, r, s = cs

complementor = preprocessor.Complementer(m, r, s, c)
factory = framefactory.CombinedFrameFactory("demo", preprocessor=complementor)
frames = []

for json, id, jpg in zip(jsons, ids, jpgs):
    j_d = framefactory.OpenPoseJsonData(json)
    c_d = framefactory.DeepSortCsvData(id)
    g_d = framefactory.DeepSortJpgData(jpg)
    frames.append(factory.create(j_d, c_d, g_d))

save_frames_to_db(frames)
