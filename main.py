import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
thepose = mp.solutions.pose

def anglz_calc(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    dot_product = np.dot(v1, v2)
    magnitude1 = np.linalg.norm(v1)
    magnitude2 = np.linalg.norm(v2)
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    angle_radians = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

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
        mp_drawing.draw_landmarks(image, results.pose_landmarks, thepose.POSE_CONNECTIONS,
                                   mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=2),
                                   mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=2))
        if results.pose_landmarks:
            p11 = results.pose_landmarks.landmark[11]
            p13 = results.pose_landmarks.landmark[13]
            p15 = results.pose_landmarks.landmark[15]
            angle = anglz_calc(p11, p13, p15)
            print(f'anglez: {angle:.2f}')
        cv2.imshow("your face or the video output", image)
        if cv2.waitKey(1) == ord('p'):
            break
    cap.release()
    cv2.destroyAllWindows()
