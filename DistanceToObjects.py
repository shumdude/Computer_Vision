import cv2
from HandTrackingModule import HandDetector
from time import sleep
import numpy as np

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(min_detection_confidence=0.8)


def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth


KNOWN_DISTANCE = 40.0
KNOWN_WIDTH = 16.0
KNOWN_PX_WIDTH = 430.0


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img, bbox = detector.find_hands(img, bbox_bool=True)
    if bbox:
        bbox_value = bbox[0]["bbox"][2]
        F = KNOWN_PX_WIDTH * KNOWN_DISTANCE / KNOWN_WIDTH
        D = (KNOWN_WIDTH * F) / bbox_value
        cv2.putText(img, "%.2fcm" % D,
                    (img.shape[1] - 400, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (0, 255, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
