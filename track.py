# #track1
# import mediapipe as mp
# import cv2
# import pyautogui,numpy as np
#
# screen_width, screen_height = pyautogui.size()
#
# # Define boundary
# boundary_margin = 0.1
# boundary_top_left = (screen_width * boundary_margin, screen_height * boundary_margin)
# boundary_bottom_right = (screen_width * (1 - boundary_margin), screen_height * (1 - boundary_margin))
#
# def map_to_screen(finger_x, finger_y, boundary_top_left, boundary_bottom_right):
#     # Map finger coordinates to screen coordinates
#     screen_x = np.interp(finger_x, [boundary_top_left[0], boundary_bottom_right[0]], [0, screen_width])
#     screen_y = np.interp(finger_y, [boundary_top_left[1], boundary_bottom_right[1]], [0, screen_height])
#     return screen_x, screen_y
#
# # Example usage
# #finger_x, finger_y = 150, 150
#
# #
# # from matplotlib import pyplot as plt
# #
# #
# mp_hand = mp.solutions.hands
# hands = mp_hand.Hands(max_num_hands=1,min_detection_confidence=0.7,min_tracking_confidence=0.7,static_image_mode=False)
# mp_draw = mp.solutions.drawing_utils
# #
# cap = cv2.VideoCapture(0)
# #
# while cap.isOpened():
#     ret,frame = cap.read()
#
#     if not ret:
#         break
#
#     frame = cv2.flip(frame,1)
#     h,w,c=frame.shape
#
#     rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#     result = hands.process(rgb)
#
#     if result.multi_hand_landmarks:
#
#         for hand_landmark in result.multi_hand_landmarks:
#
#             mp_draw.draw_landmarks(frame,hand_landmark,mp_hand.HAND_CONNECTIONS)
#             coordinates=hand_landmark.landmark[mp_hand.HandLandmark.INDEX_FINGER_TIP]
#             finger_x,finger_y=coordinates.x*w,coordinates.y*h
#             screen_x, screen_y = map_to_screen(finger_x, finger_y, boundary_top_left, boundary_bottom_right)
#             pyautogui.moveTo(screen_x, screen_y)
#
#             # Display the frame
#         cv2.imshow('Hand Tracking', frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

        # Release video capture and close windows
# cap.release()
# cv2.destroyAllWindows()

            # landmarks=[(int(lm.x*w),int(lm.y*h)) for lm in hand_landmark.landmark]
            # x_axis=[lm[0] for lm in landmarks]
            # y_axis=[lm[1] for lm in landmarks]

#             collection=[
#                 (0,1),(1,2),(2,3),(3,4),
#                 (0,5),(5,6),(6,7),(7,8),
#                 (0,9),(9,10),(10,11),(11,12),
#                 (0,13),(13,14),(14,15),(15,16),
#                 (0,17),(17,18),(18,19),(19,20),
#                 (5,9),(9,13),(13,17)
#             ]
#
#             plt.figure(figsize=(8,8))
#             plt.scatter(x_axis, y_axis, c='red')
#             for (star,end) in collection:
#                 plt.plot([x_axis[star],x_axis[end]],[y_axis[star],y_axis[end]],'blue')
#             plt.gca().invert_yaxis()
#             plt.show()
#             break
#         break
#
#
# cap.release()
# cv2.destroyAllWindows()

# cv2.imshow("tracking",frame)

            # screen_width,screen_height=pyautogui.size()
            # screen_x=int(x/w*screen_width)
            # screen_y=int(y/h*screen_height)
            # pyautogui.moveTo(screen_x,screen_y)
            #
            # distance_for_dc=((ring_finger_tip.x - thumb_finger_tip.x)**2 + (ring_finger_tip.y - thumb_finger_tip.y)**2)**0.5
            # distance_for_lc=((middle_finger_tip.x - thumb_finger_tip.x)**2 + (middle_finger_tip.y - thumb_finger_tip.y)**2)**0.5
            # current_time = time.time()
            #
            # #actions
            #
            # if distance_for_lc <= 0:
            #     if (current_time - last_click) > click_delay:
            #         event.left_click()
            #         clicking = True
            #         last_click = time.time()
            #
            # elif distance_for_dc <= 0 and not clicking:
            #     if (current_time - last_dclick) > click_delay:
            #         event.double_click()
            #         clicking = True
            #         last_click = time.time()
            #
            # else:
            #     clicking = False



# Define screen resolution


import mediapipe as mp
import cv2
import pyautogui
import numpy as np

# Get screen resolution
screen_width, screen_height = pyautogui.size()

# Define virtual boundary as a percentage of screen size
boundary_margin = 0.1  # 10% margin
boundary_top_left = (screen_width * boundary_margin, screen_height * boundary_margin)
boundary_bottom_right = (screen_width * (1 - boundary_margin), screen_height * (1 - boundary_margin))


def map_to_screen(finger_x, finger_y, boundary_top_left, boundary_bottom_right):
    # Map finger coordinates within the virtual boundary to the entire screen coordinates
    screen_x = np.interp(finger_x, [boundary_top_left[0], boundary_bottom_right[0]], [0, screen_width])
    screen_y = np.interp(finger_y, [boundary_top_left[1], boundary_bottom_right[1]], [0, screen_height])
    return screen_x, screen_y


# Initialize MediaPipe hand detection
mp_hand = mp.solutions.hands
hands = mp_hand.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7,
                      static_image_mode=False)
mp_draw = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Flip the frame horizontally for a natural feel
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convert the frame to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmark, mp_hand.HAND_CONNECTIONS)

            # Get the coordinates of the index finger tip
            coordinates = hand_landmark.landmark[mp_hand.HandLandmark.INDEX_FINGER_TIP]
            finger_x, finger_y = coordinates.x * w, coordinates.y * h

            # Map the finger coordinates to screen coordinates
            screen_x, screen_y = map_to_screen(finger_x, finger_y, boundary_top_left, boundary_bottom_right)

            # Move the cursor to the mapped screen coordinates
            pyautogui.moveTo(screen_x, screen_y)

    # Display the frame
    #cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()



