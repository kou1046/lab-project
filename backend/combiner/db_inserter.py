from __future__ import annotations
import sys

sys.path.append(".")
from typing import Sequence
from dataclasses import asdict
from api import serializers
from . import intermediate_model


def save_frames_to_db(frames: Sequence[intermediate_model.CombinedFrame]):
    serializer = serializers.FrameListSerializer(data=asdict(frames))
    serializer.is_valid()
    serializer.save()
