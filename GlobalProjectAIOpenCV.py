import cv2

from CornerRect import cornerRect
from HandTrackingModule import HandDetector
from time import sleep
import numpy as np
from pynput.keyboard import Controller

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(min_detection_confidence=0.8, max_hands=1)
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]
finalText = ""

keyboard = Controller()


def draw_all(img, button_list, clean_button):
    # for buttons
    for button in button_list:
        x, y = button.pos
        w, h = button.size
        cornerRect(img, (x, y, w, h), 20, rt=0)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 65),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

    # for CLEAN button
    x, y = clean_button.pos
    w, h = clean_button.size
    cv2.rectangle(img, (x, y), (x + w, y + h), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, clean_button.text, (x, y + h - 5), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

    return img


def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth


KNOWN_DISTANCE = 40.0
KNOWN_WIDTH = 16.0
KNOWN_PX_WIDTH = 430.0


class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text


clean_button = Button(pos=[990, 400], text='CLEAN', size=[210, 55]) # create CLEAN button
button_list = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        button_list.append(Button([100 * j + 50, 100 * i + 50], key))

DEBUG = False
SWIPE = False
UP_DOWN = False

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img, bbox = detector.find_hands(img, bbox_bool=True)
    lmList = detector.find_position(img)

    if not SWIPE and not DEBUG:
        img = draw_all(img, button_list, clean_button)

    if lmList:

        if not DEBUG:

            if bbox:
                bbox_value = bbox[0]["bbox"][2]
                F = KNOWN_PX_WIDTH * KNOWN_DISTANCE / KNOWN_WIDTH
                D = (KNOWN_WIDTH * F) / bbox_value
                cv2.putText(img, "%.2fcm" % D,
                            (img.shape[1] - 400, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            2.0, (0, 255, 0), 3)

            # Checking for up_down hand
            if lmList[0][1] - lmList[9][1] < 0:
                print('WE ARE DON\'T WORKING WITH THIS ')
                UP_DOWN = True

            '''checking swipe'''
            counter, quantity_swipe_lm = 0, 0
            for i in range(1, 5):
                for j in range(5, 8):
                    l, _, _ = detector.find_distance(j + counter, j + 1 + counter, img, draw=False)
                    # print(l)
                    if l < 20:
                        quantity_swipe_lm += 1
                counter += 4
            # print(quantity_swipe_lm)
            check_laying_flat, _, _ = detector.find_distance(0, 5, img, draw=False)
            print(check_laying_flat)
            if quantity_swipe_lm >= 8 and check_laying_flat >= 100:
                SWIPE = not SWIPE
                sleep(0.12)

            if not SWIPE and not UP_DOWN:

                '''for buttons'''
                for button in button_list:
                    x, y = button.pos
                    w, h = button.size

                    if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                        cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                        cv2.putText(img, button.text, (x + 20, y + 65),
                                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                        l, _, _ = detector.find_distance(5, 4, img, draw=False)
                        print(l)
                        if l < 30:
                            keyboard.press(button.text)
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                            cv2.putText(img, button.text, (x + 20, y + 65),
                                        cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                            finalText += button.text
                            sleep(0.1)

                '''for 'CLEAN' button'''
                x, y = clean_button.pos
                w, h = clean_button.size
                if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                    cv2.putText(img, clean_button.text, (x, y + h - 5),
                                cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    l, _, _ = detector.find_distance(5, 4, img, draw=False)
                    if l < 30:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (175, 0, 175), cv2.FILLED)
                        cv2.putText(img, clean_button.text, (x, y + h - 5), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255),
                                    4)
                        finalText = ""
                        sleep(0.1)

    if not SWIPE and not DEBUG:
        cv2.rectangle(img, (50, 350), (700, 450), (175, 0, 175), cv2.FILLED)
        cv2.putText(img, finalText, (60, 430),
                    cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    UP_DOWN = False
    cv2.imshow("Image", img)
    cv2.waitKey(1)
