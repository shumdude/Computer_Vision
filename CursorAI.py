import cv2
import numpy as np
from HandTrackingModule import HandDetector
import time, mouse

##########################
frameR = 150
smoothening = 5
wScr = 1920.0
hScr = 1080.0
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
wCam = 640
hCam = 480
cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)
detector = HandDetector(min_detection_confidence=0.8, max_hands=1)

recx = wCam - frameR
recy = hCam - frameR

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.find_hands(img)
    lmList = detector.find_position(img)
    if len(lmList) != 0:
        x1, y1 = lmList[8][0], lmList[8][1]
        fingers = detector.fingers_up(lmList)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)

        if fingers[0] == 1:
            x3 = np.interp(x1, (frameR, recx), (0, wScr)) - 20
            y3 = np.interp(y1, (frameR, recy), (0, hScr)) - 20
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            mouse.move(clocX, clocY)

            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            length, img, _ = detector.find_distance(4, 5, img, draw=False)
            if fingers[1] == 1:
                length_drag, img, _ = detector.find_distance(8, 12, img, draw=True)
                if length<15:
                    mouse.click('right')
                    time.sleep(0.08)
            else:
                if length<15:
                    mouse.click('left')
                    time.sleep(0.08)
            plocX = clocX
            plocY = clocY

    cv2.imshow("Image", img)
    cv2.waitKey(1)
