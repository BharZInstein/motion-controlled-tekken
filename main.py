import cv2
import mediapipe as mp
import numpy as np
mp_drawing=mp.solutions.drawing_utils
thepose=mp.solutions.pose
cap=cv2.VideoCapture(0)
with thepose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:

    while cap.isOpened():


        ret,frame=cap.read()
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable=False
        results=pose.process(image)

        image.flags.writeable=True
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image,results.pose_landmarks,thepose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(0,0,255),thickness=3,circle_radius=2),
        mp_drawing.DrawingSpec(color=(255,0,0),thickness=3,circle_radius=2))
        if not ret:
            continue



        cv2.imshow("Detector",image)
        
        if cv2.waitKey(1) == ord('p'):
            break


    cap.release()
    cv2.destroyAllWindows()
