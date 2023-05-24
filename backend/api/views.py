from . import serializers
from . import models
from rest_framework import viewsets


class GroupViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = serializers.GroupSerializer
    queryset = models.Group.objects.all()


class FrameViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = serializers.LightFrameSerialiser
    queryset = models.CombinedFrame.objects.all()
