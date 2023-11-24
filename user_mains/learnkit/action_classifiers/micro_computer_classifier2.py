from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2
from typing import Sequence, Callable
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms

from user_mains.learnkit import utils
from user_mains.learnkit.action_classifiers.action_classifier import ActionClassifier
from api import models

SAVE_DIR = Path(__file__).parent / "models" / "micro_computer_classifier2"


class MicroComputerClassifier2(ActionClassifier):
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
            nn.Linear(64 * 18 * 8, 512),
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

    def predict_from_persons(self, persons: Sequence[models.Person]) -> list[int]:
        assert persons
        tensor = torch.stack([val_transform(person) for person in persons]).to(self.device)
        pred_y = self(tensor)
        labels = torch.argmax(pred_y, dim=1).to("cpu").tolist()
        return labels

    def load_pretrained_data(self, device: str = "cpu"):
        return torch.load(SAVE_DIR / "epoch_300.pth", map_location=device)

    @property
    def device(self):
        return next(self.parameters()).device


train_transformer = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.RandomAffine(30, (0.1, 0.1), (0.7, 1.3)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.7),
        transforms.Resize((320, 160)),
        transforms.ToTensor(),
        transforms.Normalize(0.557, 0.231),  # Grayscale
    ]
)


def train_transform(person: models.Person) -> tuple[torch.Tensor]:
    right_hand_img = crop_hand_image(person, "right")
    left_hand_img = crop_hand_image(person, "left")

    both_hands_img = np.append(cv2.resize(right_hand_img, (160, 160)), cv2.resize(left_hand_img, (160, 160)), axis=1)

    return train_transformer(both_hands_img)


def crop_hand_image(person: models.Person, dominant: str = "right") -> np.ndarray:
    hand_range = utils.extract_hand_area(person, dominant)
    if hand_range is None:
        return np.zeros((160, 160, 3)).astype(np.uint8)

    hand_min, hand_max = hand_range

    if hand_min.x == hand_max.x or hand_min.y == hand_max.y:
        return np.zeros((160, 160, 3)).astype(np.uint8)

    screen_img = person.frame.img
    height, width, _ = screen_img.shape
    hand_img = screen_img[
        hand_min.y : hand_max.y if hand_max.y <= height else height,
        hand_min.x : hand_max.x if hand_max.x <= width else width,
    ]

    return hand_img.astype(np.uint8)


val_transformer = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((320, 160)),
        transforms.ToTensor(),
        transforms.Normalize(0.557, 0.231),  # Grayscale
    ]
)


def val_transform(person: models.Person) -> tuple[torch.Tensor]:
    right_hand_img = crop_hand_image(person, "right")
    left_hand_img = crop_hand_image(person, "left")

    both_hands_img = np.append(cv2.resize(right_hand_img, (160, 160)), cv2.resize(left_hand_img, (160, 160)), axis=1)

    return val_transformer(both_hands_img)


if __name__ == "__main__":

    class MicroComputerClassifier2Dataset(data.Dataset):
        def __init__(self, dataset: Sequence[models.Teacher], transform=Callable[[models.Person], torch.Tensor]):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index: int):
            teacher = self.dataset[index]
            label = 1 if teacher.label == 2 else 0

            if not self.transform:
                return (teacher.person,), label
            return (self.transform(teacher.person),), label

    utils.torch_fix_seed()

    inference_model = models.InferenceModel.objects.get(name="held_item")

    teachers = utils.augument_teacher_nearby_time(inference_model, 5, enable_labels=[2])
    train, test = train_test_split(teachers)

    batch_size = 400
    max_epoch = 300

    train_set = MicroComputerClassifier2Dataset(train, train_transform)
    test_set = MicroComputerClassifier2Dataset(test, val_transform)
    train_loader = data.DataLoader(train_set, batch_size)
    test_loader = data.DataLoader(test_set, batch_size)

    model = MicroComputerClassifier2()
    optim_ = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    checkpoints = [1, 50, 100, 150, 200, 230, 250, 300]

    utils.model_compile(
        model,
        train_loader,
        test_loader,
        max_epoch,
        optim_,
        criterion,
        SAVE_DIR,
        checkpoints,
    )
