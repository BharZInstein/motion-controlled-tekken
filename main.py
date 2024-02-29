import cv2
import mediapipe as mp
import numpy as np
import keyboard

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

    # Access landmarks directly using attribute access
    if results.pose_landmarks:
      poser_landmarks = results.pose_landmarks

      # Access specific landmarks using their index
      """landmark12 = poser_landmarks.landmark[12]
      landmark14 = poser_landmarks.landmark[14]
      landmark16 = poser_landmarks.landmark[16]

      h, w, _ = frame.shape
      x12, y12 = int(landmark12.x * w), int(landmark12.y * h)
      x14, y14 = int(landmark14.x * w), int(landmark14.y * h)
      x16, y16 = int(landmark16.x * w), int(landmark16.y * h)

      cv2.circle(image, (x12, y12), 5, (0, 255, 0), -1)
      cv2.circle(image, (x14, y14), 5, (0, 255, 0), -1)
      cv2.circle(image, (x16, y16), 5, (0, 255, 0), -1)"""
      

      p12 = poser_landmarks.landmark[12]
      p14 = poser_landmarks.landmark[14]
      p16 = poser_landmarks.landmark[16]
      angle = angle_calc(p12, p14, p16)
    
    if 148<angle and angle<166:
              keyboard.press_and_release("a")
              
          
    
    if cv2.waitKey(1) == ord('p'):
      break

cap.release()
cv2.destroyAllWindows()



