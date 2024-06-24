import cv2,mediapipe as mp
import util
import pyautogui,time

width,height=pyautogui.size()

mp_hands=mp.solutions.hands
hands=mp_hands.Hands(model_complexity=1,static_image_mode=False,max_num_hands=1,min_detection_confidence=0.7,min_tracking_confidence=0.7)
hand_draw=mp.solutions.drawing_utils

COOL_DOWN_TIME = 0.5
last_clik=0
def stablizier(x,y):
    if x-10<x or x or x<x+10:
        x=x
    if y-10<y or y or y<y+10:
        y=y

    return x,y

def distance_and_angle(data,list_value,index_finger):
    global last_clik
    # print(data.landmark[5])
    angle=util.get_degree(data.landmark[5],data.landmark[6],data.landmark[8])
    distance=util.get_distance([list_value[4],list_value[5]])
    x_axis=int((index_finger.x*width))
    y_axis=int((index_finger.y*height))

    if distance<50 and angle>90:
        pyautogui.moveTo(x_axis,y_axis)
        #print(x,y)
    #if util.get_degree(data.landmark[9],data.landmark[10],data.landmark[12])>90 and util.get_degree(data.landmark[5],data.landmark[6],data.landmark[8])<90:
    if util.get_distance([list_value[4],list_value[8]])<:
        current_time=time.time()
        if current_time - last_clik > COOL_DOWN_TIME:
            print('clicked')
            pyautogui.click()
            last_clik=current_time

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
# import cv2
# import mediapipe as mp
# import pyautogui
# import numpy as np
#
# # Initialize MediaPipe hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.8)
#
# # Capture video from the webcam
# cap = cv2.VideoCapture(0)
#
# # Define the boundary dimensions as a proportion of the frame size
# boundary_width_ratio = 1  # 50% of the frame width
# boundary_height_ratio =1  # 50% of the frame height
#
# # Get the screen resolution
# screen_width, screen_height = pyautogui.size()
#
# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break
#
#     frame=cv2.flip(frame,1)
#     # Get frame dimensions
#     frame_height, frame_width, _ = frame.shape
#
#     # Calculate boundary dimensions
#     boundary_width = int(frame_width * boundary_width_ratio)
#     boundary_height = int(frame_height * boundary_height_ratio)
#
#     # Define the top-left corner of the boundary
#     boundary_x = (frame_width - boundary_width) // 2
#     boundary_y = (frame_height - boundary_height) // 2
#
#     # Convert the BGR image to RGB
#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(image)
#
#     # Convert the image back to BGR
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#     # Draw the virtual boundary (green rectangle) on the image
#     cv2.rectangle(image, (boundary_x, boundary_y),
#                   (boundary_x + boundary_width, boundary_y + boundary_height),
#                   (0, 255, 0), 2)
#
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Get the coordinates of the index finger tip
#             index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#
#             # Convert normalized coordinates to pixel values
#             finger_x = int(index_finger_tip.x * frame_width)
#             finger_y = int(index_finger_tip.y * frame_height)
#
#             # Check if the finger is within the virtual boundary
#             if boundary_x <= finger_x <= boundary_x + boundary_width and boundary_y <= finger_y <= boundary_y + boundary_height:
#                 # Map the finger position to screen coordinates
#                 mapped_x = int((finger_x - boundary_x) / boundary_width * screen_width)
#                 mapped_y = int((finger_y - boundary_y) / boundary_height * screen_height)
#
#                 # Move the mouse cursor to the mapped screen coordinates
#                 pyautogui.moveTo(mapped_x, mapped_y)
#
#             # Draw the landmarks on the image
#             mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#             # Draw the finger position for visualization
#             cv2.circle(image, (finger_x, finger_y), 10, (255, 0, 0), -1)
#
#     # Display the image
#     #cv2.imshow('Hand Tracking with Boundary', image)
#
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()
