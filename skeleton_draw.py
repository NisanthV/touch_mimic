import mediapipe as mp
import cv2

from matplotlib import pyplot as plt


mp_hand = mp.solutions.hands
hands = mp_hand.Hands(max_num_hands=1,min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret,frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame,1)
    w,h,c=frame.shape

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand_landmark in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame,hand_landmark,mp_hand.HAND_CONNECTIONS)
            landmarks=[(int(lm.x*w),int(lm.y*h)) for lm in hand_landmark.landmark]
            x_axis=[lm[0] for lm in landmarks]
            y_axis=[lm[1] for lm in landmarks]
            print(x_axis,y_axis)
            plt.figure(figsize=(8,8))
            plt.scatter(x_axis,y_axis,c='red')
            plt.show()
            break
        break


cap.release()
cv2.destroyAllWindows()