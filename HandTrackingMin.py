import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

previousTime = 0
currentTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(handLmarks.landmark):
                # print(id, lm)
                height, width, channels = img.shape
                centerX, centerY = int(lm.x * width), int(lm.y * height)
                print(f'ID: {id}, X: {centerX}, Y: {centerY}')
                if id == 8:
                    cv2.circle(img, (centerX, centerY), 15, (0, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLmarks, mpHands.HAND_CONNECTIONS)

    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img, str(int(fps)), (5, 40), cv2.FONT_HERSHEY_PLAIN, 3,
                (0, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)