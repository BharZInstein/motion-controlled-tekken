import cv2
import mediapipe as mp
import directkeys

mp_drawing = mp.solutions.drawing_utils
thepose = mp.solutions.pose

def control_character(left_shoulder, right_shoulder):
    threshold = 0.1  # Adjust this threshold as needed
    if left_shoulder.x < right_shoulder.x - threshold:
        directkeys.PressKey(directkeys.Left)  # Move character left
        directkeys.ReleaseKey(directkeys.Right)
        print("Moving Left")
    elif right_shoulder.x < left_shoulder.x - threshold:
        directkeys.PressKey(directkeys.Right)  # Move character right
        directkeys.ReleaseKey(directkeys.Left)
        print("Moving Right")
    else:
        directkeys.ReleaseKey(directkeys.Left)
        directkeys.ReleaseKey(directkeys.Right)

cap = cv2.VideoCapture(0)

with thepose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            poser_landmarks = results.pose_landmarks

            left_shoulder = poser_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = poser_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]

            control_character(left_shoulder, right_shoulder)

        cv2.imshow("Character Control", image)
        if cv2.waitKey(1) == ord('p'):
            break

cap.release()
cv2.destroyAllWindows()
666666666666666666666666666666666666666666666666666666666666666666666