from __future__ import annotations
import os
import questionary
from typing import Generator, Callable
import glob
from dataclasses import asdict
from tqdm import tqdm

import db_setup
from api import serializers

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


def choice_output_by_cui():
    outputs_dirs = glob.glob("/outputs/*")
    if not outputs_dirs:
        raise FileNotFoundError("/outputs にディレクトリが見つかりません．")
    choiced_dir = questionary.select(f"{len(outputs_dirs)}個見つかりました．適用したいディレクトリを選択してください．", outputs_dirs).ask()
    if not choiced_dir:
        raise ValueError("ディレクトリを選択してください．")
    return choiced_dir


def complement_tracking():
    """
    __summary__:
        DEEPSORTの追跡不良を補完するアプリを起動するラッパー関数.
        アプリ動作中の補完情報は4つの補完ファイルに保存され, /outputs/{動画名のフォルダ}/complements/ に保存される．
    """

    choiced_dir = choice_output_by_cui()
    ids = glob.glob(f"{choiced_dir}/ID/*.csv")
    jpgs = glob.glob(f"{choiced_dir}/ID/*.jpg")
    creator = ComplementIdCreator(ids, jpgs, choiced_dir)
    creator.mainloop()


def group_data_generator(
    group_name: str, base_point: intermediate_model.KeyPointAttr = "midhip", return_length: int = 5000
) -> Generator[list[intermediate_model.CombinedFrame], None, None]:
    """
    __summary__:
        OPNEPOSEとDEEPSORTのデータを結合する処理をまとめたラッパー関数. ジェネレータなので返り値の受け取り方に注意．
        前述のcomplement_trackingで作成したディレクトリ (/outputs/{動画名のフォルダ}/complements) がある場合, 自動で読み込む.

    Args:
        group_name (str): データの名前 (自分が好きにつける).
        base_point (KeyPointAttr): 結合において，基準とする点．(デフォルトは中央腰)
        return_length: int 何フレームごとにデータを返すか. 例えば全部で20万フレームのデータを持つとメモリがクラッシュするため, この値により結果を分割し，都度保存することでメモリクラッシュを防ぐ．

    Returns:
        frames (list[CombinedFrame]): フレームごとに OPENPOSE と DEEPSORT を結合したデータ. このデータの型については submodules_aggregator/intermediate_model.py を参考に.

    Examples:
        1.
            creator = group_data_generator("demo_1")
            for data in creator:
                print(data)
            >>
            [CombinedFrame(number=1, img_path="your/path/something.jpg", people=[...]), ...]

        2.
            from submodules_aggregator.db_inserter import save_frames_to_db
            creator = create_data("demo_2", "neck", 10000)
            for data in creator:
                save_frames_to_db(data) # 10000フレーム毎にDBに登録
    """
    choiced_dir = choice_output_by_cui()

    keypoint_jsons = glob.glob(os.path.join(choiced_dir, "keypoints", "*.json"))
    id_csvs = glob.glob(os.path.join(choiced_dir, "ID", "*.csv"))
    id_jpgs = glob.glob(os.path.join(choiced_dir, "ID", "*.jpg"))

    assert len(keypoint_jsons) == len(id_csvs) == len(id_jpgs), "json, csv, jpgの数が一致しません"

    # complementsフォルダがある場合，補完を適用する
    if os.path.exists(os.path.join(choiced_dir, "complements")):
        complements_ids = [
            os.path.join(choiced_dir, "complements", file_name)
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

    framefactory = CombinedFrameFactory(group_name, base_point, preprocessor=preprocessor)

    N = len(keypoint_jsons)
    frames: list[intermediate_model.CombinedFrame] = []
    for i, (json, csv, jpg) in tqdm(enumerate(zip(keypoint_jsons, id_csvs, id_jpgs)), total=N):
        openpose_json_data = OpenPoseJsonData(json)
        deepsort_csv_data = DeepSortCsvData(csv)
        deep_jpg_data = DeepSortJpgData(jpg)
        frame = framefactory.create(openpose_json_data, deepsort_csv_data, deep_jpg_data)
        frames.append(frame)
        if len(frames) >= return_length or i + 1 == N:
            yield frames
            frames: list[intermediate_model.CombinedFrame] = []


def save_group_to_db(frames: list[intermediate_model.CombinedFrame]):
    """
    _summary_
        前述の関数のgroup_data_generatorにて結合したデータをDBに登録する関数. 登録したデータは django のORM からアクセスできる.
        DBに登録すると ORMを通じてリレーションが使えるので後々のデータ抽出が楽になる.

    Args:
        frames (list[intermediate_model.CombinedFrame]): group_data_generatorから得られるデータ.

    Examples:
        1.
            generator = group_data_generator("demo_1")
            for frames in generator:
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
    serializer.is_valid(raise_exception=True)
    serializer.save()
