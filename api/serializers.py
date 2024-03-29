import base64
from collections import OrderedDict

from django.db import transaction
from rest_framework import serializers

from .models import *


class PointSerializer(serializers.ModelSerializer):
    class Meta:
        model = Point
        fields = ["x", "y"]


class ProbabilisticPointSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProbabilisticPoint
        fields = ["x", "y", "p"]


class BoxSerializer(serializers.ModelSerializer):
    min = PointSerializer()
    max = PointSerializer()

    class Meta:
        model = BoundingBox
        fields = ["min", "max", "id"]


class KeypointSerializer(serializers.ModelSerializer):
    for p in POINT_NAMES:
        exec(f"{p} = ProbabilisticPointSerializer()")

    class Meta:
        model = KeyPoint
        fields = POINT_NAMES


class PersonSerializer(serializers.ModelSerializer):
    box = BoxSerializer()
    keypoint = KeypointSerializer()

    class Meta:
        model = Person
        fields = ["box", "keypoint"]


class LightPersonSerializer(serializers.ModelSerializer):
    frameNum = serializers.IntegerField(source="frame.number")
    group = serializers.CharField(source="frame.group.name")

    class Meta:
        model = Person
        fields = ["id", "frameNum", "group"]


class GroupSerializer(serializers.ModelSerializer):
    class Meta:
        model = Group
        fields = ["name"]
        extra_kwargs = {"name": {"validators": []}}  # 　これがないと既にDBにあるgroupの名前で（別のフレームのデータを）登録しようとしたとき，is_validでひっかかる


class FrameSerializer(serializers.ModelSerializer):
    people = PersonSerializer(many=True)
    group = GroupSerializer()

    class Meta:
        model = Frame
        fields = ["group", "img_path", "people", "number"]


class LightFrameSerialiser(serializers.ModelSerializer):
    group = GroupSerializer()
    people = LightPersonSerializer(many=True)

    class Meta:
        model = Frame
        fields = ["group", "img_path", "people", "number"]


class FrameListSerializer(serializers.ListSerializer):
    child = FrameSerializer()

    def create(self, validated_data_list: list[OrderedDict]):
        new_people = []
        new_frames = []
        new_boxes = []
        new_keypoints = []
        new_probilistic_points = []
        new_points = []
        new_groups = []
        for validated_data in validated_data_list:
            people = validated_data.pop("people")
            new_group = Group(**validated_data["group"])
            validated_data["group"] = new_group
            new_frame = Frame(**validated_data)
            new_groups.append(new_group)
            new_frames.append(new_frame)
            for person in people:
                for name, probabilistic_point in person["keypoint"].items():
                    new_probilistic_point = ProbabilisticPoint(**probabilistic_point, group=new_group)
                    person["keypoint"][name] = new_probilistic_point
                    new_probilistic_points.append(new_probilistic_point)
                for range_name in ("min", "max"):
                    point = person["box"][range_name]
                    new_point = Point(**point, group=new_group)
                    person["box"][range_name] = new_point
                    new_points.append(new_point)
                new_box = BoundingBox(**person["box"])
                new_keypoint = KeyPoint(**person["keypoint"])

                new_person = Person(box=new_box, keypoint=new_keypoint, frame=new_frame, group=new_group)
                new_boxes.append(new_box)
                new_keypoints.append(new_keypoint)
                new_people.append(new_person)
        with transaction.atomic():
            Group.objects.bulk_create(new_groups, ignore_conflicts=True)
            Point.objects.bulk_create(new_points)
            ProbabilisticPoint.objects.bulk_create(new_probilistic_points)
            KeyPoint.objects.bulk_create(new_keypoints)
            BoundingBox.objects.bulk_create(new_boxes)
            Frame.objects.bulk_create(new_frames)
            Person.objects.bulk_create(new_people)
        return new_frames


class ReadOnlyFrameSerializer(serializers.ModelSerializer):
    device = serializers.CharField(source="device.id")

    class Meta:
        model = Frame
        fields = ["group", "frame", "id", "device", "people"]


class ClickSerializer(serializers.ModelSerializer):
    class Meta:
        model = MouseClick
        fields = "__all__"


class ReleaseSerializer(serializers.ModelSerializer):
    class Meta:
        model = MouseRelease
        fields = "__all__"


class DragSerializer(serializers.ModelSerializer):
    click = ClickSerializer()
    release = ReleaseSerializer()

    class Meta:
        model = MouseDrag
        fields = "__all__"


class MousePosSerializer(serializers.ModelSerializer):
    class Meta:
        model = MousePos
        fields = ["time", "x", "y"]


class MouseReleaseSerializer(serializers.ModelSerializer):
    class Meta:
        model = MouseRelease
        fields = ["time", "x", "y", "frame"]


class MouseClickSerializer(serializers.ModelSerializer):
    class Meta:
        model = MouseClick
        fields = ["time", "x", "y", "frame"]


class MouseDragSerializer(serializers.ModelSerializer):
    click = MouseClickSerializer()
    release = MouseReleaseSerializer()
    group = GroupSerializer()

    class Meta:
        model = MouseDrag
        fields = ["id", "click", "release", "group", "person"]


class DeviceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Device
        fields = ["id"]


class InferenceModelSerializer(serializers.ModelSerializer):
    labelDescription = serializers.CharField(source="label_description")

    class Meta:
        model = InferenceModel
        fields = ["id", "name", "labelDescription"]


class TeacherSerializer(serializers.ModelSerializer):
    class Meta:
        model = Teacher
        fields = ["person", "label", "model"]

    def to_representation(self, instance):
        data = {
            "label": instance.label,
            "person": LightPersonSerializer(instance.person).data,
        }
        return data
