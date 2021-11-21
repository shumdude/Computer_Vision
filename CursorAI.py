import cv2, time, mouse, numpy
from HandTrackingModule import HandDetector

#######PARAMETERS#######
frame_rectangle = 150
smoothening = 5
width_screen = 1920.0
height_screen = 1080.0
old_clocX, old_clocY = 0, 0
clocX, clocY = 0, 0
width_camera = 640
height_camera = 480
click_length = 16
#########################

cap = cv2.VideoCapture(1)
cap.set(3, width_camera)
cap.set(4, height_camera)
detector = HandDetector(min_detection_confidence=0.8, max_hands=1)
rectangleX = width_camera - frame_rectangle
rectangleY = height_camera - frame_rectangle

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.find_hands(img)
    landmark_list = detector.find_position(img)
    if len(landmark_list) != 0:
        x1, y1 = landmark_list[8][0], landmark_list[8][1]
        fingers = detector.fingers_up(landmark_list)
        cv2.rectangle(img,
                      (frame_rectangle, frame_rectangle),
                      (width_camera - frame_rectangle, height_camera - frame_rectangle),
                      (255, 0, 255), 2)

        # Checking the index finger
        if fingers[0] == 1:
            x3 = numpy.interp(x1, (frame_rectangle, rectangleX), (0, width_screen)) - 20
            y3 = numpy.interp(y1, (frame_rectangle, rectangleY), (0, height_screen)) - 20
            clocX = old_clocX + (x3 - old_clocX) / smoothening
            clocY = old_clocY + (y3 - old_clocY) / smoothening

            # Move mouse
            mouse.move(clocX, clocY)

            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            length, img, _ = detector.find_distance(4, 5, img, draw=False)

            #  Checking middle finger
            if fingers[1] == 1:
                length_drag, img, landmarks_list = detector.find_distance(12, 8, img, draw=False)
                cv2.circle(img, (landmarks_list[0], landmarks_list[1]), 15, (255, 0, 255), cv2.FILLED)
                # Right click
                if length < click_length:
                    mouse.click('right')
                    time.sleep(0.08)

                # Drag objects
                if length_drag < 27:
                    mouse.press()
                else:
                    mouse.release()

            else:
                # Left click
                if length < click_length:
                    mouse.click('left')
                    time.sleep(0.08)

            old_clocX = clocX
            old_clocY = clocY

    cv2.imshow("CursorAI", img)
    cv2.waitKey(1)
