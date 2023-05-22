from mypkg.submodules_aggregator import utils
from api import models
import cv2

person_id_1_data = models.Person.objects.filter(frame__group="demo_1", box__id=1)
for person_id_1 in person_id_1_data:
    cv2.imshow("img", person_id_1.visualize())
    cv2.waitKey(1)
