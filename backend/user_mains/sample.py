from tkinter.filedialog import askdirectory
import glob
from combiner import complementidcreator, preprocessor, framefactory
import cv2

base_dir = askdirectory(initialdir="/outputs")
jsons = glob.glob(f"{base_dir}/keypoints/*.json")
ids = glob.glob(f"{base_dir}/ID/*.csv")
jpgs = glob.glob(f"{base_dir}/ID/*.jpg")
cs = glob.glob(f"{base_dir}/*.json")
c, m, r, s = cs

complementor = preprocessor.Complementer(m, r, s, c)
factory = framefactory.CombinedFrameFactory("demo", preprocessor=complementor)

for json, id, jpg in zip(jsons, ids, jpgs):
    j_d = framefactory.OpenPoseJsonData(json)
    c_d = framefactory.DeepSortCsvData(id)
    g_d = framefactory.DeepSortJpgData(jpg)
    frame = factory.create(j_d, c_d, g_d)
    cv2.imshow("img", frame.visualize())
    cv2.waitKey(1)
