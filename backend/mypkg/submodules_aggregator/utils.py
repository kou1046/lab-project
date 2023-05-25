import os
from tkinter.filedialog import askdirectory
from typing import Sequence
import glob
from dataclasses import asdict
from tqdm import tqdm

import db_setup
from api import serializers
from api import models

from .complementidcreator import (
    replace_filename,
    monitor_filename,
    create_filename,
    stopresume_filename,
)
from . import intermediate_model
from .complementidcreator import ComplementIdCreator
from .framefactory import (
    OpenPoseJsonData,
    DeepSortCsvData,
    DeepSortJpgData,
    CombinedFrameFactory,
)
from .preprocessor import Complementer


def complement_tracking():
    """
    __summary__:
        DEEPSORTの追跡不良を補完するアプリを起動するラッパー関数.
        アプリ動作中の補完情報は4つの補完ファイルに保存され, /outputs/{動画名のフォルダ}/complements/ に保存される．
    """

    base_dir = askdirectory(initialdir="/outputs")
    ids = glob.glob(f"{base_dir}/ID/*.csv")
    jpgs = glob.glob(f"{base_dir}/ID/*.jpg")
    creator = ComplementIdCreator(ids, jpgs, base_dir)
    creator.mainloop()


def create_group_data(
    group_name: str,
    base_point: intermediate_model.KeyPointAttr = "midhip",
) -> list[intermediate_model.CombinedFrame]:
    """
    __summary__:
        OPNEPOSEとDEEPSORTのデータを結合する処理をまとめたラッパー関数.
        前述のcomplement_trackingで作成したディレクトリ (/outputs/{動画名のフォルダ}/complements) がある場合, 自動で読み込む.

    Args:
        group_name (str): データの名前 (自分が好きにつける).
        base_point (KeyPointAttr): 結合において，基準とする点．(デフォルトは中央腰)

    Returns:
        frames (list[CombinedFrame]): フレームごとに OPENPOSE と DEEPSORT を結合したデータ. このデータの型については submodules_aggregator/intermediate_model.py を参考に.

    Examples:
        1.
            data = create_group_data("demo_1")
            print(data)

            >>
            [CombinedFrame(number=1, img_path="your/path/something.jpg", people=[...]), ...]

        2.
            from submodules_aggregator.db_inserter import save_frames_to_db
            data = create_data("demo_2", "neck")
            save_frames_to_db(data) #DBに登録
    """
    base_dir = askdirectory(initialdir="/outputs")
    if not base_dir:
        exit()
    keypoint_jsons = glob.glob(os.path.join(base_dir, "keypoints", "*.json"))
    id_csvs = glob.glob(os.path.join(base_dir, "ID", "*.csv"))
    id_jpgs = glob.glob(os.path.join(base_dir, "ID", "*.jpg"))

    assert (
        len(keypoint_jsons) == len(id_csvs) == len(id_jpgs)
    ), "json, csv, jpgの数が一致しません"

    # complementsフォルダがある場合，補完を適用する
    if os.path.exists(os.path.join(base_dir, "complements")):
        complements_ids = [
            os.path.join(base_dir, "complements", file_name)
            for file_name in (
                monitor_filename,
                replace_filename,
                stopresume_filename,
                create_filename,
            )
        ]
        preprocessor = Complementer(*complements_ids)
    else:
        preprocessor = None

    framefactory = CombinedFrameFactory(
        group_name, base_point, preprocessor=preprocessor
    )

    frames: list[intermediate_model.CombinedFrame] = []
    for json, csv, jpg in tqdm(
        zip(keypoint_jsons, id_csvs, id_jpgs), total=len(keypoint_jsons)
    ):
        openpose_json_data = OpenPoseJsonData(json)
        deepsort_csv_data = DeepSortCsvData(csv)
        deep_jpg_data = DeepSortJpgData(jpg)
        frame = framefactory.create(
            openpose_json_data, deepsort_csv_data, deep_jpg_data
        )
        frames.append(frame)
    return frames


def save_group_to_db(frames: list[intermediate_model.CombinedFrame]):
    """
    _summary_
        前述の関数のcreate_group_dataにて結合したデータをDBに登録する関数. 登録したデータは django のORM からアクセスできる.
        DBに登録すると ORMを通じてリレーションが使えるので後々のデータ抽出が楽になる.

    Args:
        frames (list[intermediate_model.CombinedFrame]): create_group_dataの返り値.

    Examples:
        1.
            frames = create_group_data("demo_1")
            save_frames_to_db(frames)

            import db_setup
            from api import models

            group = models.Group.objects.get(name="demo_1")

            #↑このgroupにデータが詰まっている. (group.frames.all(), group.frames.all().first().people.all() 等,
            # groupから必要な情報を取れるように, djangoのORMの使い方を調べるとよい. モデルは api/models.py を参照に. )

            print(group)

            >>
                Group object(name="demo_1")
    """

    serializer = serializers.FrameListSerializer(data=[asdict(ins) for ins in frames])
    serializer.is_valid()
    serializer.save()
