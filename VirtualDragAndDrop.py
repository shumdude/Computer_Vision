import time

import cv2
from HandTrackingModule import HandDetector
import numpy as np
previous_time = 0
current_time = 0
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(min_detection_confidence=0.8)
colorR = (0, 0, 255)

centerX, centerY, width, height = 100, 100, 200, 200


def iter_rects(rect_list, cursor):
    for rect in rect_list:
        rect.update(cursor)

class DragRectangle():
    def __init__(self, position_center, size=[200, 200]):
        self.position_center = position_center
        self.size = size

    def update(self, cursor):
        centerX = self.position_center[0]
        centerY = self.position_center[1]
        width, height = self.size
        if centerX - width // 2 < cursor[0] < centerX + width // 2 \
                and centerY - height // 2 < cursor[1] < centerY + height // 2:
            self.position_center = cursor

rect_list = []
for i in range(0,5):
    rect = DragRectangle(position_center=[i*250+150, 150])
    rect_list.append(rect)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.find_hands(img=img)
    landmark_list = detector.find_position(img, draw=False)
    if landmark_list:
        length, _, _ = detector.find_distance(p1=8, p2=4, img=img, draw=False)
        # print(length)
        if length < 40:
            cursor = landmark_list[8]
            # if centerX - width // 2 < cursor[1] < centerX + width // 2 and centerY - height // 2 < cursor[
            #     2] < centerY + height // 2:
            #     id, centerX, centerY = cursor
            iter_rects(rect_list, cursor)
            # rect.update(cursor)

    for rect in rect_list:
        centerX = rect.position_center[0]
        centerY = rect.position_center[1]
        width = rect.size[0]
        height = rect.size[1]

        cv2.rectangle(img, (centerX - width // 2, centerY - height // 2), (centerX + width // 2, centerY + height // 2),
                      colorR, cv2.FILLED)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
