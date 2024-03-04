import cv2
import mediapipe as mp
import numpy as np
import directkeys
import time
import threading

mp_drawing = mp.solutions.drawing_utils
thepose = mp.solutions.pose

def angle_calc(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    dot_product = np.dot(v1, v2)
    magnitude1 = np.linalg.norm(v1)
    magnitude2 = np.linalg.norm(v2)
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

def calculate_mean(points):
    if len(points) == 0:
        return None
    mean_point = np.mean(points, axis=0)
    return mean_point

def key_press_release(key):
    directkeys.PressKey(key)
    time.sleep(0.1)  # Adjust the duration if needed
    directkeys.ReleaseKey(key)

cap = cv2.VideoCapture(0)

with thepose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            poser_landmarks = results.pose_landmarks

            p12 = poser_landmarks.landmark[12]
            p14 = poser_landmarks.landmark[14]
            p16 = poser_landmarks.landmark[16]
            angle = angle_calc(p12, p14, p16)
            p24 = poser_landmarks.landmark[24]
            p26 = poser_landmarks.landmark[26]
            p28 = poser_landmarks.landmark[28]
            anglel = angle_calc(p24, p26, p28)

            if 148 < angle < 166:
                threading.Thread(target=key_press_release, args=(0x1E,)).start()
            if 75 < angle <= 90:
                threading.Thread(target=key_press_release, args=(0x48,)).start()
            if 40 < anglel < 90:
                threading.Thread(target=key_press_release, args=(0x1F,)).start()

        if cv2.waitKey(1) == ord('p'):
            break

cap.release()
cv2.destroyAllWindows()
