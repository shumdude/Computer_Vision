import math

import cv2
import mediapipe as mp
import time


class HandDetector():

    def __init__(self, mode=False, max_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.max_hands,
                                        min_detection_confidence=self.min_detection_confidence,
                                        min_tracking_confidence=self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True, bbox_bool=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        h, w, c = img.shape
        bbox_list = []
        if self.results.multi_hand_landmarks:
            for handLmarks in self.results.multi_hand_landmarks:
                my_hand = {}
                xList = []
                yList = []
                for id, lm in enumerate(handLmarks.landmark):
                    px, py = int(lm.x * w), int(lm.y * h)
                    xList.append(px)
                    yList.append(py)
                # bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)
                my_hand["bbox"] = bbox
                my_hand["center"] = (cx, cy)

                if draw:
                    self.mpDraw.draw_landmarks(img, handLmarks, self.mpHands.HAND_CONNECTIONS)
                    if bbox_bool:
                        cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                      (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                      (255, 0, 255), 2)
                bbox_list.append(my_hand)

        if bbox_bool:
            return img, bbox_list
        return img

    def find_position(self, img, hand_number=0, draw=True):
        self.landmark_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(my_hand.landmark):
                # print(id, lm)
                height, width, channels = img.shape[0], img.shape[1], img.shape[2]
                centerX, centerY = int(lm.x * width), int(lm.y * height)
                # print(f'ID: {id}, X: {centerX}, Y: {centerY}')
                self.landmark_list.append([centerX, centerY])
                if draw:
                    cv2.circle(img, (centerX, centerY), 5, (255, 0, 255), cv2.FILLED)

        return self.landmark_list

    def find_distance(self, p1, p2, img, draw=True):
        x1 = self.landmark_list[p1][0]
        y1 = self.landmark_list[p1][1]
        x2, y2 = self.landmark_list[p2][0], self.landmark_list[p2][1]
        centerX, centerY = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (centerX, centerY), 15, (255, 0, 255), cv2.FILLED)

        first_arg = x2 - x1
        second_arg = y2 - y1
        length = math.hypot(first_arg, second_arg)
        return length, img, [x1, y1, x2, y2, centerX, centerY]

    def fingers_ip(self, lmList):
        if self.results.multi_hand_landmarks:
            fingers = []
            for id in range(1, 5):
                if lmList[self.tipIds[id]][1] < lmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            return fingers


def main():
    previous_time = 0
    current_time = 0
    cap = cv2.VideoCapture(1)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.find_hands(img=img)
        landmark_list = detector.find_position(img=img)
        if len(landmark_list) != 0:
            print(landmark_list[8])

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(img, str(int(fps)), (5, 40), cv2.FONT_HERSHEY_PLAIN, 3,
                    (0, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
