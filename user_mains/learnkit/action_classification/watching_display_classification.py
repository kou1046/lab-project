from __future__ import annotations

from typing import Sequence, Callable
from pathlib import Path

from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms

import db_setup
from api import models
from user_mains.learnkit import utils


class WatchingDisplayClassifier(nn.Module):
    def __init__(self):
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
            nn.Linear(1344, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        y = self.block_1(x)
        y = self.block_2(y)
        y = y.view(x.shape[0], -1)
        y = self.block_3(y)
        return y

    def predict_from_people(self, people: list[models.Person], device="cpu") -> list[int]:  # 推論
        imgs: list[torch.Tensor] = []
        if not people:
            return []
        for person in people:
            imgs.append(val_transform(person))
        return torch.argmax(self(torch.stack(imgs).to(device)), dim=1).to(device).tolist()


def train_transform(
    person: models.Person,
) -> tuple[torch.Tensor, torch.Tensor]:
    face_range = utils.extract_face_area(person)

    if face_range is None:  # 切り抜きに必要な関節が欠けているなら，ダミーを返す．
        return torch.zeros((1, 82, 148))

    min_, max_ = face_range

    transformer = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.RandomAffine(5, (0.05, 0.05), (0.8, 1.2)),
            transforms.RandomHorizontalFlip(0.3),
            transforms.ColorJitter(brightness=0.7),
            transforms.Resize((82, 148)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.25),
        ]
    )

    screen_img = person.frame.img
    height, width, _ = screen_img.shape
    person_target_img = screen_img[
        min_.y : max_.y if max_.y <= height else height, min_.x : max_.x if max_.x <= width else width
    ]

    img = transformer(person_target_img)
    return img


def val_transform(person: models.Person):
    face_range = utils.extract_face_area(person)

    if face_range is None:  # 切り抜きに必要な関節が欠けているなら，ダミーを返す．
        return torch.zeros((1, 82, 148))

    min_, max_ = face_range

    transformer = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((82, 148)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.25),
        ]
    )

    screen_img = person.frame.img
    height, width, _ = screen_img.shape
    person_target_img = screen_img[
        min_.y : max_.y if max_.y <= height else height, min_.x : max_.x if max_.x <= width else width
    ]

    img = transformer(person_target_img)
    return img


def train_classifier(train_dataset: ProgrammingClassifierDataset, test_dataset: ProgrammingClassifierDataset):
    model = WatchingDisplayClassifier()
    optim_ = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    batch_size = 16
    train_loader = data.DataLoader(train_dataset, batch_size)
    test_loader = data.DataLoader(test_dataset, batch_size)
    checkpoints = [50, 100, 150, 200, 230, 250, 300]

    utils.model_compile(
        model,
        train_loader,
        test_loader,
        300,
        optim_,
        criterion,
        Path("./submodules/user_mains/learnkit/models/watching_classifier"),
        checkpoints,
    )


if __name__ == "__main__":

    class ProgrammingClassifierDataset(data.Dataset):
        def __init__(self, dataset: Sequence[models.Teacher], transform=Callable[[models.Person], torch.Tensor]):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index: int):
            teacher = self.dataset[index]
            label = 1 if teacher.label == 1 else 0

            if not self.transform:
                return teacher.person, label
            return self.transform(teacher.person), label

    inference_model = models.InferenceModel.objects.get(name="held_item")

    teachers = utils.augument_teacher_nearby_time(inference_model)
    train, test = train_test_split(list(teachers))

    train_dataset = ProgrammingClassifierDataset(train, transform=train_transform)
    test_dataset = ProgrammingClassifierDataset(test, transform=val_transform)

    train_classifier(train_dataset, test_dataset)
