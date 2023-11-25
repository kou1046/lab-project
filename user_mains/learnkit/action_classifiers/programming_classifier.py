from __future__ import annotations

from pathlib import Path
from typing import Sequence, Callable
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms

from user_mains.learnkit import utils
from api import models

SAVE_DIR = Path(__file__).parent / "models" / "programming_classifier"


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

    def forward(self, x):
        y = self.block_1(x)
        y = self.block_2(y)
        y = y.view(x.shape[0], -1)
        y = self.block_3(y)
        return y

    def predict_from_persons(self, persons: Sequence[models.Person]) -> list[int]:
        assert persons
        img_tensor = torch.stack([val_transform(person) for person in persons]).to(self.device)
        pred_y = self(img_tensor)
        labels = torch.argmax(pred_y, dim=1).to("cpu").tolist()
        return labels

    def load_pretrained_data(self, device: str = "cpu"):
        return torch.load(SAVE_DIR / "epoch_300.pth", map_location=device)


def train_transform(person: models.Person) -> tuple[torch.Tensor]:
    dominant = "right"

    if person.group.name == "ube_G5_2022" and person.box.id == 5:
        dominant = "left"

    hand_range = utils.extract_hand_area(person, dominant)

    if hand_range is None:  # 切り抜きに必要な関節が欠けているなら，ダミーを返す．
        return torch.zeros((1, 160, 160))

    min_point, max_point = hand_range

    if min_point.x == max_point.x or min_point.y == max_point.y:
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
    dominant = "right"

    if person.group.name == "ube_G5_2022" and person.box.id == 5:
        dominant = "left"

    hand_range = utils.extract_hand_area(person, dominant)

    if hand_range is None:  # 切り抜きに必要な関節が欠けているなら，ダミーを返す．
        return torch.zeros((1, 160, 160))

    min_point, max_point = hand_range

    if min_point.x == max_point.x or min_point.y == max_point.y:
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
                return (teacher.person,), label
            return (self.transform(teacher.person),), label

    utils.torch_fix_seed()

    inference_model = models.InferenceModel.objects.get(name="held_item")

    teachers = utils.augument_teacher_nearby_time(inference_model, 3, enable_labels=[1])
    train, test = train_test_split(teachers)

    batch_size = 256
    max_epoch = 300

    train_set = ProgrammingClassifierDataset(train, train_transform)
    test_set = ProgrammingClassifierDataset(test, val_transform)
    train_loader = data.DataLoader(train_set, batch_size)
    test_loader = data.DataLoader(test_set, batch_size)

    model = ProgrammingClassifier()
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
