import cv2
import mediapipe as mp
import time

capture = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

prev_time = 0
curr_time =0

while True:
    success, img = capture.read()
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(img_RGB)
    if res.multi_hand_landmarks:
        for handLandmark in res.multi_hand_landmarks:
            for id, lm in enumerate(handLandmark.landmark):
                print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id==0:
                   cv2.circle(img, (cx, cy), 10, (100, 72, 164), cv2.FILLED)
            mp_draw.draw_landmarks(img, handLandmark, mp_hands.HAND_CONNECTIONS)

    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    prev_time=curr_time

    cv2.putText(img, str(int(fps)), (10, 65), cv2.FONT_HERSHEY_PLAIN, 3, (100, 72, 164), 5)



    cv2.imshow("Video Capture", img)
    cv2.waitKey(1)