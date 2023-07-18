import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import threading
import queue
import time
from data_processing import *
from data_collection import *
from matplotlib.animation import FuncAnimation
from mediapipe.framework.formats import landmark_pb2
from mpl_toolkits.mplot3d import Axes3D

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


_landmarks_list, _landmark_connections, _captured_landmark_df = None, None, create_xyz_landmark_df()
_status, _data_cache = False, queue.Queue()

with mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5) as hands, mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        
        ret, frame = cap.read()
        
        image, _landmarks_list, _landmark_connections = detect_upperbody(frame, hands, pose)
    
        if _landmarks_list:
            _data_cache.put(_landmarks_list)
            mp_drawing.draw_landmarks(image, _landmarks_list, _landmark_connections)
        
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            
            break
        

cap.release()
cv2.destroyAllWindows()

df = create_xyz_landmark_df()
df = df_entry_from_queue_NLL(df = df, data_cache = _data_cache, centered = True)
create_csv(df = df)
