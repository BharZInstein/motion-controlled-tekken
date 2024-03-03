import cv2
import mediapipe as mp
import numpy as np
import math
import keyboard
import threading

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)
cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Hand Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
current_direction = None
running = True

def get_angle(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    angle_rad = math.atan(slope)
    angle_deg = math.degrees(angle_rad)
    if x2 < x1:
         angle_deg += 180
    angle_deg = angle_deg % 360
    return angle_deg

def get_direction(angle_deg):
    if 45 <= angle_deg < 135:
        return 'up'
    elif 135 <= angle_deg < 225:
        return 'left'
    elif 225 <= angle_deg < 315:
        return 'down'
    else:
        return 'right'

def handle_keys(direction):
    global current_direction
    if direction != current_direction:
        current_direction = direction
        keyboard.press(current_direction)
    else:
        keyboard.release(current_direction)

def process_frames():
    global current_direction
    global running
    while running:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            for landmark in hand_landmarks.landmark:
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            angle = get_angle(hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y,
                              hand_landmarks.landmark[6].x, hand_landmarks.landmark[6].y)
            direction = get_direction(angle)
            handle_keys(direction)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

frame_thread = threading.Thread(target=process_frames)
frame_thread.start()

while running:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

cap.release()
cv2.destroyAllWindows()