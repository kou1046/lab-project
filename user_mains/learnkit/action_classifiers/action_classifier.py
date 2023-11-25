from abc import abstractmethod

from typing import Sequence

import torch
from torch import nn

import db_setup
from api import models


class ActionClassifier(nn.Module):
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @abstractmethod
    def load_pretrained_data(self, device: str = "cpu"):
        ...

    @abstractmethod
    def predict_from_persons(self, persons: Sequence[models.Person]) -> list[int]:
        ...
