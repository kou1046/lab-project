from __future__ import annotations

import os
from torch.utils import data
from typing import Sequence, Literal
import random
from pathlib import Path
import numpy as np
import db_setup
from api import models
import torch
from torch import optim
from torch import nn
from tqdm import tqdm

from submodules.deepsort_openpose.api.domain.points.point import Point
from submodules.deepsort_openpose.api import domain


def torch_fix_seed(seed=42):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def model_compile(
    model: nn.Module,
    train_loader: data.DataLoader,
    val_loader: data.DataLoader,
    max_epoch: int,
    optim: optim.SGD,
    criterion: nn.Module,
    save_dir: Path,
    checkpoints: Sequence[int] | None = None,
) -> None:
    """
    毎回学習する際ののひな型を書くのが面倒なので関数にしたもの．モデルの入力が画像のみの時なら使える.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    os.makedirs(save_dir, exist_ok=True)
    checkpoints = [max_epoch] if checkpoints is None else checkpoints

    train_accs = []
    test_accs = []

    for epoch in range(1, max_epoch + 1):
        sum_acc = 0
        model.train()
        print(f"epoch: {epoch}/{max_epoch}")
        for imgs, t in tqdm(train_loader):
            imgs = imgs.to(device)
            t = t.to(device)

            pred_y = model(imgs)
            loss = criterion(pred_y, t)
            model.zero_grad()
            loss.backward()
            optim.step()
            sum_acc += torch.sum(t == torch.argmax(pred_y, dim=1))

        print(f"train acc:{sum_acc/len(train_loader.dataset)}")
        train_accs.append(float(sum_acc / len(train_loader.dataset)))

        sum_acc = 0
        model.eval()
        with torch.no_grad():
            for imgs, t in val_loader:
                imgs = imgs.to(device)
                t = t.to(device)

                pred_y = model(imgs)
                sum_acc += torch.sum(t == torch.argmax(pred_y, dim=1))
            print(f"test acc:{sum_acc/len(val_loader.dataset)}")
            test_accs.append(float(sum_acc / len(val_loader.dataset)))

            if epoch in checkpoints:
                ts = []
                preds_ys = []
                for imgs, t in val_loader:
                    imgs = imgs.to(device)
                    t = t.to("cpu").tolist()
                    ts.extend(t)

                    preds_ys.extend(torch.argmax(model(imgs), dim=1).to("cpu").tolist())

                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "train_accs": train_accs,
                        "test_accs": test_accs,
                        "test_pred_y": preds_ys,
                        "test_true_y": t,
                    },
                    save_dir / f"epoch_{epoch}.pth",
                )


def extract_hand_area(
    person: models.Person, dominant: Literal["right", "left"] = "right"
) -> tuple[Point, Point] | None:
    neck = person.keypoint.neck
    shoulder = person.keypoint.r_shoulder if dominant == "right" else person.keypoint.l_shoulder
    wrist = person.keypoint.r_wrist if dominant == "right" else person.keypoint.l_wrist

    if not neck.p or not shoulder.p or not wrist.p:
        return None

    distance = neck.distance_to(shoulder)

    xmin = wrist.x - distance
    xmax = wrist.x + distance
    ymin = wrist.y - distance
    ymax = wrist.y + distance

    return Point(int(xmin), int(ymin)), Point(int(xmax), int(ymax))


def augument_teacher_nearby_time(inference_model: models.InferenceModel, interval_frame: int = 5):
    teachers: list[models.Teacher] = list(inference_model.teachers.all())
    augumented_teachers: list[models.Teacher] = []

    for teacher in teachers:
        frame_number = teacher.person.frame.number
        persons = models.Person.objects.filter(
            group__name=teacher.person.group.name,
            box__id=teacher.person.box.id,
            frame__number__gt=frame_number - interval_frame,
            frame__number__lt=frame_number + interval_frame,
        )

        tmp_teachers = [models.Teacher(person=person, label=teacher.label, model=inference_model) for person in persons]
        augumented_teachers.extend(tmp_teachers)

    return augumented_teachers


def extract_face_area(person: models.Person) -> tuple[Point, Point] | None:
    r_eye = person.keypoint.r_eye
    l_eye = person.keypoint.l_eye
    nose = person.keypoint.nose
    r_ear = person.keypoint.r_ear
    l_ear = person.keypoint.l_ear

    if not r_eye.p or not l_eye.p or not nose.p or not r_ear.p or not l_ear.p:
        return None

    eye_center_point = Point(
        (person.keypoint.l_eye.x + person.keypoint.r_eye.x) / 2, (person.keypoint.l_eye.y + person.keypoint.r_eye.y) / 2
    )

    dis_between_nose_eye = nose.distance_to(eye_center_point)

    ymin = int(eye_center_point.y - 2 * dis_between_nose_eye)
    ymax = int(eye_center_point.y + 2 * dis_between_nose_eye)
    xmin = int(min([r_ear.x, l_ear.x])) - int(dis_between_nose_eye)
    xmax = int(max([r_ear.x, l_ear.x])) + int(dis_between_nose_eye)

    return Point(xmin, ymin), Point(xmax, ymax)


def preprocess_keypoint(keypoint: models.KeyPoint, base_point: domain.KeyPointAttr) -> torch.Tensor:
    base_point: models.ProbabilisticPoint = getattr(keypoint, base_point)

    xs = [point.x for point in keypoint.get_all_points()]
    ys = [point.y for point in keypoint.get_all_points()]

    # 基準点の相対座標に変換
    relative_xs = [x - base_point.x for x in xs]
    relative_ys = [y - base_point.y for y in ys]

    max_x = max(relative_xs)
    max_y = max(relative_ys)

    # 正規化
    normalize_xs = [x / max_x if max_x else 0.0 for x in relative_xs]
    normalize_ys = [y / max_y if max_y else 0.0 for y in relative_ys]

    tensor = torch.Tensor([normalize_xs, normalize_ys]).T

    return tensor
