#track1
# import mediapipe as mp
# import cv2
#
# from matplotlib import pyplot as plt
#
#
# mp_hand = mp.solutions.hands
# hands = mp_hand.Hands(max_num_hands=1,min_detection_confidence=0.7)
# mp_draw = mp.solutions.drawing_utils
#
# cap = cv2.VideoCapture(0)
#
# while cap.isOpened():
#     ret,frame = cap.read()
#
#     if not ret:
#         break
#
#     frame = cv2.flip(frame,1)
#     w,h,c=frame.shape
#
#     rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#     result = hands.process(rgb)
#
#     if result.multi_hand_landmarks:
#
#         for hand_landmark in result.multi_hand_landmarks:
#
#             mp_draw.draw_landmarks(frame,hand_landmark,mp_hand.HAND_CONNECTIONS)
#             landmarks=[(int(lm.x*w),int(lm.y*h)) for lm in hand_landmark.landmark]
#             x_axis=[lm[0] for lm in landmarks]
#             y_axis=[lm[1] for lm in landmarks]
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