import cv2,mediapipe as mp
import util
import pyautogui

width,height=pyautogui.size()

mp_hands=mp.solutions.hands
hands=mp_hands.Hands(model_complexity=1,static_image_mode=False,max_num_hands=1,min_detection_confidence=0.7,min_tracking_confidence=0.7)
hand_draw=mp.solutions.drawing_utils


def distance_and_angle(data,list_value,index_finger):
    # print(data.landmark[5])
    angle=util.get_degree(data.landmark[5],data.landmark[6],data.landmark[8])
    distance=util.get_distance([list_value[4],list_value[5]])
    x_axis=int((index_finger.x*width))
    y_axis=int((index_finger.y*height))

    if distance<50 and angle>90:
        pyautogui.moveTo(x_axis,y_axis)

def main():

    cam=cv2.VideoCapture(0)

    try:
        while cam.isOpened():

            ret,frame = cam.read()

            if not ret:
                break
            frame=cv2.flip(frame,1)
            frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            processed=hands.process(frameRGB)
            land_marks=[]

            if processed.multi_hand_landmarks:
                hand_landmark=processed.multi_hand_landmarks[0]
                hand_draw.draw_landmarks(frame,hand_landmark,mp_hands.HAND_CONNECTIONS)

                index_finger=hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                for lm in hand_landmark.landmark:
                    land_marks.append((lm.x,lm.y))

                distance_and_angle(hand_landmark,land_marks,index_finger)


            cv2.imshow('cam',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()