import cv2
import mediapipe as mp
import numpy as np
import keyboard

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize MediaPipe Pose model
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to check if the person has jumped
def check_jump(landmarks):
    # Get the y-coordinate of relevant landmarks
    left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
    
    # Calculate the average y-coordinate of shoulders and hips
    avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
    avg_hip_y = (left_hip_y + right_hip_y) / 2
    
    # If the average shoulder y-coordinate is higher than the average hip y-coordinate, it indicates a jump
    if avg_shoulder_y < avg_hip_y:
        return True
    else:
        return False

# Function to simulate pressing the up arrow key
def jump():
    keyboard.press('up')
    keyboard.release('up')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect pose landmarks
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        # Visualize the pose landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Check for a jump
        if check_jump(results.pose_landmarks.landmark):
            jump()
    
    cv2.imshow('MediaPipe Pose Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pose.close()
cap.release()
cv2.destroyAllWindows()
