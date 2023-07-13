from __future__ import annotations
import os
import glob


class FrameElementDirectory:
    KEYPOINT_OUTPUT_DIR_NAME = "keypoints"
    DEEPSORT_OUTPUT_DIR_NAME = "ID"

    def __init__(self, directory_path: str):
        if not os.path.exists(os.path.join(directory_path, self.KEYPOINT_OUTPUT_DIR_NAME)):
            ValueError("参照するディレクトリが間違っている")
        if not os.path.exists(os.path.join(directory_path, self.DEEPSORT_OUTPUT_DIR_NAME)):
            ValueError("参照するディレクトリが間違っている")
        keypoint_json_paths = glob.glob(os.path.join(directory_path, self.KEYPOINT_OUTPUT_DIR_NAME, "*.json"))
        deepsort_csv_paths = glob.glob(os.path.join(directory_path, self.DEEPSORT_OUTPUT_DIR_NAME, "*.csv"))
        deepsort_jpg_paths = glob.glob(os.path.join(directory_path, self.DEEPSORT_OUTPUT_DIR_NAME, "*.jpg"))
        assert (
            len(keypoint_json_paths) == len(deepsort_csv_paths) == len(deepsort_jpg_paths)
        ), "OPENPOSE, DEEPSORTの出力の数が異なっている"

        self.total = len(keypoint_json_paths)
        self.keypoint_json_gen = iter(keypoint_json_paths)
        self.deepsort_csv_gen = iter(deepsort_csv_paths)
        self.deepsort_jpg_paths = iter(deepsort_jpg_paths)

    def __iter__(self):
        return self

    def __next__(self):
        return (next(self.keypoint_json_gen), next(self.deepsort_csv_gen), next(self.deepsort_jpg_paths))
