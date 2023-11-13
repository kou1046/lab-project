from __future__ import annotations

import os
from typing import Sequence
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms


from user_mains.learnkit import utils
from api import models


class ProgrammingClassifier(nn.Module):  # ProgrammingClassifier
    def __init__(self, pretrained_model_path: str = None):
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2, 2),
        )

        self.block_3 = nn.Sequential(
            nn.Linear(4096, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
        )

        if pretrained_model_path is not None:
            self.load_state_dict(
                torch.load(
                    pretrained_model_path,
                )["model_state_dict"]
            )
            self.eval()

    def forward(self, x):
        y = self.block_1(x)
        y = self.block_2(y)
        y = y.view(x.shape[0], -1)
        y = self.block_3(y)
        return y

    def predict_from_people(self, people: Sequence[models.Person], device: str = "cpu") -> list[int, list[float]]:  # 推論
        imgs = []
        if not people:
            return [], []
        for person in people:
            imgs.append(val_transform(person).to(device))
        result = self(torch.stack(imgs))
        ts = torch.argmax(result, dim=1).tolist()
        probs = torch.softmax(result, dim=1).tolist()
        if not any(ts) and people[0].frame.group.name == "G3":  # Group3のみマウスが映らないことが多いので特定の処理を追加する
            r_wrist_ys = [person.keypoints.r_wrist.y for person in people]
            r_wrist_max_y = max(r_wrist_ys)
            if r_wrist_max_y > 670:
                predict_index = r_wrist_ys.index(r_wrist_max_y)
                ts[predict_index] = 1
        return ts, probs


def get_extracted_range(person: models.Person) -> tuple[models.Point, models.Point] | tuple[None, None]:
    # if person.box.id == 5 and person.frame.group.name == "G5":  ## !!!lazyな変更　後で最適化
    #     dominant = "left"
    # else:
    #     dominant = "right"
    dominant = "right"
    side_ = "r_" if dominant == "right" else "l_"
    required_points: list[models.ProbabilisticPoint] = [
        getattr(person.keypoint, attr) for attr in ("neck", side_ + "shoulder", side_ + "wrist")
    ]
    neck, shoulder, wrist = required_points
    if not neck.p or not shoulder.p or not wrist.p:
        return None, None, None, None

    distance = neck.distance_to(shoulder)

    xmin = wrist.x - distance
    xmax = wrist.x + distance
    ymin = wrist.y - distance
    ymax = wrist.y + distance

    return models.Point(xmin, ymin), models.Point(xmax, ymax)


def train_transform(person: models.Person) -> tuple[torch.Tensor]:
    min_point, max_point = get_extracted_range(person)
    if min_point is None or min_point.x == max_point.x or min_point.y == max_point.y:  # 切り抜きに必要な関節が欠けているなら，ダミーを返す．
        return torch.zeros((1, 160, 160))

    transformer = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.RandomAffine(30, (0.1, 0.1), (0.7, 1.3)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.7),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(0.557, 0.231),  # Grayscale
        ]
    )

    screen_img = person.frame.img
    height, width, _ = screen_img.shape
    person_target_img = screen_img[
        min_point.y : max_point.y if max_point.y <= height else height,
        min_point.x : max_point.x if max_point.x <= width else width,
    ]
    return transformer(person_target_img)


def val_transform(person: models.Person) -> tuple[torch.Tensor]:
    min_point, max_point = get_extracted_range(person)
    if min_point is None or min_point.x == max_point.x or min_point.y == max_point.y:  # 切り抜きに必要な関節が欠けているなら，ダミーを返す．
        return torch.zeros((1, 160, 160))

    transformer = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(0.557, 0.231),  # Grayscale
        ]
    )

    screen_img = person.frame.img
    height, width, _ = screen_img.shape
    person_target_img = screen_img[
        min_point.y : max_point.y if max_point.y <= height else height,
        min_point.x : max_point.x if max_point.x <= width else width,
    ]
    return transformer(person_target_img)


def collate_fn(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    imgs = []
    labels = []
    for sample, t in batch:
        imgs.append(sample)
        labels.append(t)
    return torch.stack(imgs).to("cuda"), torch.Tensor(labels).long().to("cuda")


if __name__ == "__main__":
    estimator = models.InferenceModel.objects.get(name="Programming")

    teachers = estimator.teachers
    train, test = train_test_split(list(teachers.all()))

    batch_size = 16
    max_epoch = 500

    train_set = utils.TeacherDataset(train, train_transform)
    test_set = utils.TeacherDataset(test, val_transform)
    train_loader = data.DataLoader(train_set, batch_size, collate_fn=collate_fn)
    test_loader = data.DataLoader(test_set, batch_size, collate_fn=collate_fn)

    model = ProgrammingClassifier().to("cuda")
    optim_ = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    checkpoints = [100, 150, 200, 230, 250, 270, 300, 400, 500]

    utils.model_compile(
        model,
        train_loader,
        test_loader,
        max_epoch,
        optim_,
        criterion,
        os.path.join(os.path.dirname(__file__), "models", "programming_classifier"),
        checkpoints,
    )
