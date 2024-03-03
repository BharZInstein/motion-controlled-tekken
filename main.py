import cv2
import mediapipe as mp
import numpy as np
import directkeys
import time

mp_drawing = mp.solutions.drawing_utils
thepose = mp.solutions.pose

def angle_calc(p1, p2):
    v1 = np.array([p1.x, p1.y])
    v2 = np.array([p2.x, p2.y])
    dot_product = np.dot(v1, v2)
    magnitude1 = np.linalg.norm(v1)
    magnitude2 = np.linalg.norm(v2)
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

def control_character(angle):
    if angle > 0:
        directkeys.PressKey(directkeys.Right)
        directkeys.ReleaseKey(directkeys.Left)
    else:
        directkeys.PressKey(directkeys.Left)
        directkeys.ReleaseKey(directkeys.Right)

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

            left_shoulder = poser_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = poser_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]

            angle = angle_calc(left_shoulder, right_shoulder)

            control_character(angle)

        cv2.imshow("Your Face", image)
        if cv2.waitKey(1) == ord('p'):
            break

cap.release()
cv2.destroyAllWindows()
