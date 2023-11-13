from __future__ import annotations
import os
from pathlib import Path
import questionary
from tqdm import tqdm

from submodules.deepsort_openpose.api.applications.complement_tracking_application import (
    Complementer,
    monitor_filename,
    replace_filename,
    stopresume_filename,
    create_filename,
)
from submodules.deepsort_openpose.api.domain import Frame, KeyPoint, KeyPointAttr
from submodules.deepsort_openpose.api.domain.frames.frame_factory import FrameFactory, FrameElementDirectory
from user_mains.domain.groups.igroup_repository import IGroupRepository
from user_mains.domain.groups import Group


class GroupRegisterCLIApplication:
    REGISTER_CHUNK_FRAME_NUM = 3000

    def __init__(self, group_repository: IGroupRepository):
        self.group_repository = group_repository

    def register(self):
        output_dir = Path("./submodules/deepsort_openpose/outputs")
        dirs = [str(dir_) for dir_ in output_dir.glob("*")]
        if not dirs:
            print(f"{str(output_dir)}にopenpose, deepsortの出力が見つかりませんでした. ")
            exit()

        choiced_dirs: list[str] = questionary.checkbox(f"{len(dirs)}個見つかりました．適用するディレクトリを選択してください．", dirs).ask()
        names: list[str] = []
        base_points: list[KeyPointAttr] = []
        for choiced_dir in choiced_dirs:
            name = questionary.text(f"{choiced_dir} Name?", default=Path(choiced_dir).name).ask()
            base_point = questionary.select(f"{choiced_dir} base keypoint?", KeyPoint.NAMES, default="midhip").ask()
            names.append(name)
            base_points.append(base_point)

        for choiced_dir, name, base_point in zip(choiced_dirs, names, base_points):
            gen = self._generate_frame_data(choiced_dir, base_point)
            for chunk_frames in gen:
                group = Group(name, chunk_frames)
                self.group_repository.save(group)

    def _generate_frame_data(self, choiced_dir, base_point):
        complement_dir = os.path.join(choiced_dir, "complements")
        if os.path.exists(complement_dir):
            f = lambda filename: os.path.join(complement_dir, filename)
            preprocessor = Complementer(
                f(monitor_filename),
                f(replace_filename),
                f(stopresume_filename),
                f(create_filename),
            )
        else:
            preprocessor = None
        element_directory = FrameElementDirectory(Path(choiced_dir))
        frame_factory = FrameFactory(element_directory, base_point, preprocessor)
        frames: list[Frame] = []
        for i, frame in tqdm(enumerate(frame_factory, 1), total=element_directory.total):
            frames.append(frame)
            if len(frames) >= self.REGISTER_CHUNK_FRAME_NUM or i == element_directory.total:
                yield frames
                frames: list[Frame] = []
