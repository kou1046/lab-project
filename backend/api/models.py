from __future__ import annotations

import base64
import datetime
import uuid

import cv2
import numpy as np
from django.db import models


class UUIDModel(models.Model):
    class Meta:
        abstract = True

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)


class NonNegativeFloatField(models.FloatField):
    def to_python(self, value):
        value = super().to_python(value)
        if value is not None and value < 0:
            return 0
        return value


class AbstractPoint(UUIDModel):
    class Meta:
        abstract = True

    x = NonNegativeFloatField()
    y = NonNegativeFloatField()

    def distance_to(self, other: Point):
        return np.linalg.norm(np.array([self.x - other.x, self.y - other.y]))

    def angle(self, a_point: Point, c_point: Point):
        vec_1 = np.array([a_point.x - self.x, c_point.y - c_point.y])
        vec_2 = np.array([c_point.x - self.x, c_point.y - c_point.y])
        return np.arccos((np.dot(vec_1, vec_2)) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))) * 180 / np.pi


class Point(AbstractPoint):
    class Meta:
        db_table = "point"


class ProbabilisticPoint(AbstractPoint):
    class Meta:
        db_table = "probabilistic_point"

    p = NonNegativeFloatField(default=100.0)


class BoundingBox(models.Model):
    class Meta:
        db_table = "boundingbox"

    pk_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    id = models.IntegerField()
    min = models.OneToOneField(Point, models.CASCADE, related_name="+")
    max = models.OneToOneField(Point, models.CASCADE, related_name="+")

    def center(self) -> Point:
        center_x = (self.max.x - self.min.x) / 2.0 + float(self.min.x)
        center_y = (self.max.y - self.min.y) / 2.0 + float(self.min.y)
        return Point(x=center_x, y=center_y)

    def contains(self, point: Point) -> bool:
        return point.x >= self.min.x and point.x <= self.max.x and point.y >= self.min.y and point.y <= self.max.y


POINT_NAMES = [
    "nose",
    "neck",
    "r_shoulder",
    "r_elbow",
    "r_wrist",
    "l_shoulder",
    "l_elbow",
    "l_wrist",
    "midhip",
    "r_hip",
    "r_knee",
    "r_ankle",
    "l_hip",
    "l_knee",
    "l_ankle",
    "r_eye",
    "l_eye",
    "r_ear",
    "l_ear",
    "l_bigtoe",
    "l_smalltoe",
    "l_heel",
    "r_bigtoe",
    "r_smalltoe",
    "r_hell",
]


class KeyPoint(UUIDModel):
    class Meta:
        db_table = "keypoint"
        default_related_name = "+"

    nose = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    neck = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    r_shoulder = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    r_elbow = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    r_wrist = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    l_shoulder = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    l_elbow = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    l_wrist = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    midhip = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    r_hip = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    r_knee = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    r_ankle = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    l_hip = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    l_knee = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    l_ankle = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    r_eye = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    l_eye = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    r_ear = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    l_ear = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    l_bigtoe = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    l_smalltoe = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    l_heel = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    r_bigtoe = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    r_smalltoe = models.OneToOneField(ProbabilisticPoint, models.CASCADE)
    r_hell = models.OneToOneField(ProbabilisticPoint, models.CASCADE)

    def get_all_points(self) -> list[Point]:
        return [getattr(self, name) for name in POINT_NAMES]

    @property
    def face_angle(self) -> float | None:
        return self.neck.angle(self.r_eye, self.l_eye)

    @property
    def neck_angle(self) -> float | None:
        return self.neck.angle(self.nose, self.midhip)

    @property
    def r_elbow_angle(self) -> float | None:
        return self.r_elbow.angle(self.r_shoulder, self.r_wrist)

    @property
    def l_elbow_angle(self) -> float | None:
        return self.l_elbow.angle(self.l_shoulder, self.l_wrist)

    @property
    def r_shoulder_angle(self) -> float | None:
        return self.r_shoulder.angle(self.neck, self.r_elbow)

    @property
    def l_shoulder_angle(self) -> float | None:
        return self.l_shoulder.angle(self.neck, self.l_elbow)

    @property
    def shoulder_dis(self) -> float | None:
        return self.r_shoulder.distance_to(self.l_shoulder)

    @property
    def eye_dis(self) -> float | None:
        return self.r_eye.distance_to(self.l_eye)

    @property
    def elbow_dis(self) -> float | None:
        return self.r_elbow.distance_to(self.l_elbow)


class Group(models.Model):
    class Meta:
        db_table = "group"

    name = models.CharField(primary_key=True, max_length=50)


class CombinedFrame(UUIDModel):
    class Meta:
        db_table = "frame"
        constraints = [models.UniqueConstraint(fields=["number", "group"], name="group_frame_num_unique")]

    number = models.IntegerField()
    img_path = models.CharField(max_length=100)
    group = models.ForeignKey(Group, models.CASCADE, related_name="frames")
    # people (逆参照)

    @property
    def img(self) -> np.ndarray:
        return cv2.imread(self.img_path)

    @property
    def img_base64(self) -> str:
        ret, dst_data = cv2.imencode(".jpg", self.img)
        return base64.b64encode(dst_data)

    def get_visualized_image(
        self,
        draw_keypoint: bool = False,
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
            if draw_keypoint:
                for point in person.keypoint.get_all_points():
                    cv2.circle(
                        img,
                        (int(point.x), int(point.y)),
                        point_radius,
                        color,
                        thickness,
                    )
        return img


class MousePos(UUIDModel):
    class Meta:
        db_table = "mouse_position"
        constraints = [models.UniqueConstraint(fields=["group", "time"], name="unique_mouse_position")]

    group = models.ForeignKey(Group, models.CASCADE, related_query_name="mouse_postions")
    time = models.TimeField()
    x = models.IntegerField()
    y = models.IntegerField()


class MouseClick(UUIDModel):
    class Meta:
        db_table = "click"

    time = models.TimeField()
    x = models.IntegerField()
    y = models.IntegerField()
    frame = models.OneToOneField(CombinedFrame, models.CASCADE, null=True)


class MouseRelease(UUIDModel):
    class Meta:
        db_table = "release"

    time = models.TimeField()
    x = models.IntegerField()
    y = models.IntegerField()
    frame = models.OneToOneField(CombinedFrame, models.CASCADE, null=True)


class Device(UUIDModel):
    class Meta:
        db_table = "device"

    screenshot_path = models.CharField(max_length=100)
    frame = models.OneToOneField(CombinedFrame, models.CASCADE)
    group = models.ForeignKey(Group, models.CASCADE, "devices")
    mouse_pos = models.ForeignKey(MousePos, models.PROTECT)
    mouse_click = models.OneToOneField(MouseClick, models.PROTECT, null=True)
    mouse_release = models.OneToOneField(MouseRelease, models.PROTECT, null=True)

    @property
    def screenshot(self) -> np.ndarray:
        return cv2.imread(self.screenshot_path)

    @property
    def screenshot_base64(self):
        ret, dst_data = cv2.imencode(".jpg", self.screenshot)
        return base64.b64encode(dst_data)

    @property
    def drawn_screenshot(self) -> np.ndarray:
        ss = self.screenshot
        color = (0, 0, 0) if self.mouse_click is None and self.mouse_release is None else (0, 0, 255)
        cv2.circle(ss, (self.mouse_pos.x, self.mouse_pos.y), 5, color, 10)
        return ss

    @property
    def drawn_screenshot_base64(self) -> np.ndarray:
        ret, dst_data = cv2.imencode(".jpg", self.drawn_screenshot)
        return base64.b64encode(dst_data)


class Person(UUIDModel):
    class Meta:
        db_table = "person"

    keypoint = models.OneToOneField(KeyPoint, models.CASCADE, related_name="+")
    box = models.OneToOneField(BoundingBox, models.CASCADE, related_name="+")
    frame = models.ForeignKey(CombinedFrame, models.CASCADE, related_name="people")

    @property
    def img(self) -> np.ndarray:
        frame_img = self.frame.img
        screen_height, screen_width, _ = frame_img.shape
        img = frame_img[
            int(self.box.min.y) : (int(self.box.max.y) if self.box.max.y <= screen_height else screen_height),
            int(self.box.min.x) : (int(self.box.max.x) if self.box.max.x <= screen_width else screen_width),
        ]
        return img

    @property
    def img_base64(self):
        ret, dst_data = cv2.imencode(".jpg", self.img)
        return base64.b64encode(dst_data)

    def visualize_screen_img(
        self,
        draw_keypoints: bool = False,
        color: tuple[int, int, int] = (0, 0, 0),
        point_radius: int = 5,
        thickness: int = 3,
        isbase64=False,
    ) -> np.ndarray:
        img_copy = self.frame.img
        cv2.rectangle(
            img_copy,
            (self.box.min.x, self.box.min.y),
            (self.box.max.x, self.box.max.y),
            color,
            thickness,
        )
        if draw_keypoints:
            for point in self.keypoint.get_all_points():
                cv2.circle(
                    img_copy,
                    (int(point.x), int(point.y)),
                    point_radius,
                    color,
                    thickness,
                )
        if isbase64:
            ret, dst_data = cv2.imencode(".jpg", img_copy)
            return base64.b64encode(dst_data)
        return img_copy

    def visualize(
        self,
        color: tuple[int, int, int] = (0, 0, 0),
        point_radius: int = 5,
        thickness: int = 3,
        isbase64=False,
    ) -> np.ndarray:
        img_copy = self.img.copy()
        for point in self.keypoint.get_all_points():
            cv2.circle(
                img_copy,
                (int(point.x) - int(self.box.min.x), int(point.y) - int(self.box.min.y)),
                point_radius,
                color,
                thickness,
            )
        if isbase64:
            ret, dst_data = cv2.imencode(".jpg", img_copy)
            return base64.b64encode(dst_data)
        return img_copy


class MouseDrag(UUIDModel):
    class Meta:
        db_table = "drag"
        constraints = [models.UniqueConstraint(fields=["click", "group", "release"], name="unique_drag")]

    click = models.OneToOneField(MouseClick, on_delete=models.CASCADE)
    release = models.OneToOneField(MouseRelease, on_delete=models.CASCADE)
    group = models.ForeignKey(Group, on_delete=models.CASCADE, related_name="drags")
    person = models.OneToOneField(Person, models.CASCADE, null=True)

    @property
    def distance(self) -> float:
        return np.linalg.norm(np.array([self.click.x - self.release.x, self.click.y - self.release.y]))

    @property
    def time(self) -> float:
        d = datetime.datetime.now().date()
        td = datetime.datetime.combine(d, self.release.time) - datetime.datetime.combine(d, self.click.time)
        return td.total_seconds()


class InferenceModel(UUIDModel):
    class Meta:
        db_table = "inference_model"

    name = models.CharField(max_length=50, unique=True)
    label_description = models.CharField(max_length=100)
    model_path = models.CharField(max_length=50, null=True)

    def __str__(self):
        return self.name


class Teacher(models.Model):
    class Meta:
        db_table = "teacher"
        constraints = [models.UniqueConstraint(fields=["person", "model"], name="unique_teacher")]

    person = models.ForeignKey(Person, models.CASCADE, related_name="teachers")
    label = models.IntegerField(default=0)
    model = models.ForeignKey(InferenceModel, on_delete=models.CASCADE, related_name="teachers")


class Action(models.Model):
    class Meta:
        db_table = "action"

    person = models.OneToOneField(Person, models.CASCADE, related_name="action")
    is_programming = models.BooleanField()
    is_having_pen = models.BooleanField()
    is_watching_display = models.BooleanField()
