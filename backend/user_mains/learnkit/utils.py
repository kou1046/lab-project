import os
from torch.utils import data
from typing import TypeVar, Generic, Sequence, Callable
import db_setup
from api import models
import torch
from torch import optim
from torch import nn
from tqdm import tqdm

T = TypeVar("T")


class TeacherDataset(data.Dataset, Generic[T]):
    def __init__(self, teachers: Sequence[models.Teacher], transform: Callable[[models.Person], T] | None = None):
        self.teachers = teachers
        self.transform = transform

    def __len__(self):
        return len(self.teachers)

    def __getitem__(self, index: int):
        teacher = self.teachers[index]
        if self.transform is not None:
            return self.transform(teacher.person), teacher.label
        return teacher.person.img, teacher.label


def model_compile(
    model: nn.Module,
    train_loader: data.DataLoader,
    val_loader: data.DataLoader,
    max_epoch: int,
    optim: optim.SGD,
    criterion: nn.Module,
    save_dir: str,
    checkpoints: Sequence[int] | None = None,
) -> None:
    """
    毎回学習する際ののひな型を書くのが面倒なので関数にしたもの．モデルの入力が画像のみの時なら使える.
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoints = [max_epoch] if checkpoints is None else checkpoints

    train_accs = []
    test_accs = []
    for epoch in range(1, max_epoch + 1):
        sum_acc = 0
        model.train()
        for imgs, t in tqdm(train_loader, total=len(train_loader)):
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
                pred_y = model(imgs)
                sum_acc += torch.sum(t == torch.argmax(pred_y, dim=1))
            print(f"test acc:{sum_acc/len(val_loader.dataset)} epoch {epoch}/{max_epoch} done.")
            test_accs.append(float(sum_acc / len(val_loader.dataset)))
            if epoch in checkpoints:
                ts = []
                preds_ys = []
                for imgs, t in val_loader:
                    ts += t.tolist()
                    preds_ys += torch.argmax(model(imgs), dim=1).tolist()
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "train_accs": train_accs,
                        "test_accs": test_accs,
                        "test_pred_y": preds_ys,
                        "test_true_y": ts,
                    },
                    os.path.join(os.path.join(save_dir, f"epoch_{epoch}_model.pth")),
                )
