from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from ..people import Person


@dataclass(frozen=True)
class Frame:
    people: list[Person]
    img_path: str
    number: int

    def __eq__(self, other: Frame):
        return self.number == other.number

    def __hash__(self):
        return hash(self.number)

    @property
    def img(self):
        return cv2.imread(self.img_path)

    def contains_person(self):
        return any(self.people)

    def person_ids(self) -> set[int]:
        return {person.id for person in self.people}

    def visualize(
        self,
        draw_keypoints: bool = False,
        color: tuple[int, int, int] = (0, 0, 0),
        point_radius: int = 5,
        thickness: int = 3,
    ) -> np.ndarray:
        img = self.img.copy()
        for person in self.people:
            cv2.putText(
                img,
                f"ID:{person.box.id}",
                (person.box.min.x, person.box.min.y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                thickness,
            )
            cv2.rectangle(
                img,
                (person.box.min.x, person.box.min.y),
                (person.box.max.x, person.box.max.y),
                color,
                thickness,
            )
            if draw_keypoints:
                for point in person.keypoint.get_all_points():
                    cv2.circle(
                        img,
                        (int(point.x), int(point.y)),
                        point_radius,
                        color,
                        thickness,
                    )
        return img
