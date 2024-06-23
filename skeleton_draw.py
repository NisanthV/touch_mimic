import mediapipe as mp
import cv2
# import pyautogui
import time

import win32api
import win32con,math


class EventManagement:

    def left_click(self):
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0,0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        print('left click')
        return None
    def double_click(self):
        print('dclicked')
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        print('dclicked')
        return None

mp_hand = mp.solutions.hands
hands = mp_hand.Hands(max_num_hands=1,model_complexity=1,min_detection_confidence=0.7,static_image_mode=False,min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
clicking = False
last_click = 0
last_dclick = 0
click_delay = 0.2
event=EventManagement()
alpha = 0.2
scr_x,scr_y = 0,0

def calculate_distance(lm1, lm2):
    return math.sqrt((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2)


while cap.isOpened():
    ret,frame = cap.read()

    if not ret:
        break
    #frame = cv2.convertScaleAbs(frame, alpha=2, beta=10)
    frame = cv2.flip(frame,1)
    h,w,c=frame.shape

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand_landmark in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame,hand_landmark,mp_hand.HAND_CONNECTIONS)
            index_finger_tip = hand_landmark.landmark[mp_hand.HandLandmark.INDEX_FINGER_TIP]
            thumb_finger_tip = hand_landmark.landmark[mp_hand.HandLandmark.THUMB_TIP]
            middle_finger_tip = hand_landmark.landmark[mp_hand.HandLandmark.MIDDLE_FINGER_TIP]
            ring_finger_tip = hand_landmark.landmark[mp_hand.HandLandmark.RING_FINGER_TIP]

            x=int(index_finger_tip.x*w)
            y=int(index_finger_tip.y*h)

            #cv2.circle(frame,(x,y),10,(0,255,0),cv2.FILLED)

            screen_width, screen_height = win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)
            screen_x = int(index_finger_tip.x * screen_width)
            screen_y = int(index_finger_tip.y * screen_height)

            scr_x = int(alpha*screen_x + (1-alpha)*screen_x)
            scr_y = int(alpha*screen_y + (1-alpha)*screen_y)



            win32api.SetCursorPos((scr_x, scr_y))

            # distance_for_lc = ((middle_finger_tip.x - thumb_finger_tip.x) ** 2 + (
            #             middle_finger_tip.y - thumb_finger_tip.y) ** 2) ** 0.5
            # distance_for_dc = ((ring_finger_tip.x - thumb_finger_tip.x) ** 2 + (
            #             ring_finger_tip.y - thumb_finger_tip.y) ** 2) ** 0.5

            distance_for_lc = calculate_distance(middle_finger_tip, thumb_finger_tip)
            distance_for_dc = calculate_distance(ring_finger_tip, thumb_finger_tip)

            current_time = time.time()

            if distance_for_lc < 0.03:
                if (current_time - last_click) > click_delay:
                    #event.left_click()
                    last_click = current_time

            elif distance_for_dc < 0.03:
                if (current_time - last_dclick) > click_delay:
                    #event.double_click()
                    last_dclick = current_time

            clicking = (distance_for_lc < 0.03 or distance_for_dc < 0.03)

            cv2.imshow('tracking',frame)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break


cap.release()
cv2.destroyAllWindows()