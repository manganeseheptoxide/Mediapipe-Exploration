import mediapipe as mp
import cv2
import pandas as pd
from mediapipe.framework.formats import landmark_pb2

NormLandmark = mp.tasks.components.containers.NormalizedLandmark
NormLandmarkList = landmark_pb2.NormalizedLandmarkList

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

useful_pose_landmarks_index = [0, 7, 8, 9, 10, 11, 12, 13, 14]

_hand_connections_1 = [pair for pair in mp_hands.HAND_CONNECTIONS]
_hand_connections_2 = [(pair[0] + 21, pair[1] + 21) for pair in mp_hands.HAND_CONNECTIONS]
_pose_connections_nohands = [(0, 1), (0, 2), (1, 2),
                             (0, 3), (0, 4), (3, 4),
                             (5, 6), (5, 7), (6, 8),
                             (1, 3), (2, 4)]
_pose_connections_1hand = [(pair[0] + 21, pair[1] + 21) for pair in _pose_connections_nohands]
_pose_connections_2hands = [(pair[0] + 42, pair[1] + 42) for pair in _pose_connections_nohands]

def get_connection_list(num_hands: int = 2, pose: bool = True):
    if num_hands == 0 and pose:
        out = _pose_connections_nohands
    elif num_hands == 1:
        out = _hand_connections_1 + _pose_connections_1hand if pose else _hand_connections_1
    elif num_hands == 2:
        out = _hand_connections_1 + _hand_connections_2 + _pose_connections_2hands + [(0, 50), (21, 49)] if pose else _hand_connections_1 + _hand_connections_2
    else:
        out = []
    return out

def connections(num_hands: int = 0, classification: list = [], pose: bool = False):
    global _hand_connections_1, _hand_connections_2, _pose_connections_nohands, _pose_connections_1hand, _pose_connections_2hands
    
    # Creates a list of tuples of landmark connections based on the detected data

    if num_hands == 0 and pose:
        out = _pose_connections_nohands
    elif num_hands == 1:
        out = _hand_connections_1 + _pose_connections_1hand if pose else _hand_connections_1
    elif num_hands == 2:
        out = _hand_connections_1 + _hand_connections_2 + _pose_connections_2hands if pose else _hand_connections_1 + _hand_connections_2
    else:
        out = []

    if num_hands != 0 and pose and classification:
        if num_hands == 1:
            out += [(0, 29)] if classification[0] == 'left' else [(0, 28)]
        elif num_hands == 2:
            out += [(0, 50), (21, 49)] # if classification[0] == 'left' else [(0, 49), (21, 50)]
        else:
            pass

    return out

def detect_upperbody(frame, hands, pose, only_52: bool = False, centered: bool = False):

    # modified detection using hands and pose detection
    # returns the image, landmarks, and the appropriate connections

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    image.flags.writeable = False
    hands_results = hands.process(image)
    pose_results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    container = []

    num_hands_detected = 0 if bool(hands_results.multi_hand_landmarks) == False else len(hands_results.multi_hand_landmarks)
    hand_classifications = [side.classification[0].label.lower() for side in hands_results.multi_handedness] if bool(hands_results.multi_handedness) else []
    pose_detected = bool(pose_results.pose_landmarks)

    if hands_results.multi_hand_landmarks:
        if hand_classifications[0] == 'left':
            for i, landmarks in enumerate(hands_results.multi_hand_landmarks):
                container = container + list(landmarks.landmark)
        else:
            for i, landmarks in enumerate(hands_results.multi_hand_landmarks):
                container = list(landmarks.landmark) + container
    
    if pose_results.pose_landmarks:
        for index in useful_pose_landmarks_index:
            container.append(pose_results.pose_landmarks.landmark[index])

        #### center landmark coords
        # this will be a reference point when processing the data for model training and visualization

        x = (pose_results.pose_landmarks.landmark[11].x + pose_results.pose_landmarks.landmark[12].x)/2
        y = (pose_results.pose_landmarks.landmark[11].y + pose_results.pose_landmarks.landmark[12].y)/2
        z = (pose_results.pose_landmarks.landmark[11].z + pose_results.pose_landmarks.landmark[12].z)/2
        visibility = (pose_results.pose_landmarks.landmark[11].visibility + pose_results.pose_landmarks.landmark[12].visibility)/2

        # center_landmark = NormLandmark(x = x, y = y, z = z)
        # container.append(center_landmark)

        # when creating an instance of a NormalizedLandmarkList with center_landmark appended to the iterable, it gives the error below:
        # TypeError: Parameter to MergeFrom() must be instance of same class: expected mediapipe.NormalizedLandmark got NormalizedLandmark.

        center_landmark = pose_results.pose_landmarks.landmark[15]
        center_landmark.x = x
        center_landmark.y = y
        center_landmark.z = z
        center_landmark.visibility = visibility

        container.append(center_landmark)

        # this will be a workaround for now because of the error that i do not know how to fix,.
    if centered :
        if not pose_results.pose_landmarks:
            raise ValueError('Since there are no pose landmarks, there is no center point you fuckwit')
        for landmark in container:
            landmark.x -= container[-1].x
            landmark.y -= container[-1].y
            landmark.z -= container[-1].z

    if only_52 and container:
        useful_landmarks = NormLandmarkList(landmark = container) if len(container) == 52 else NormLandmarkList()
    else:
        useful_landmarks = NormLandmarkList(landmark = container) if container else NormLandmarkList()

    connection_data = connections(num_hands = num_hands_detected, classification = hand_classifications, pose = pose_detected) if useful_landmarks.landmark else []
    
    # image, NormalizedLandmarkList, and list
    return image, useful_landmarks, connection_data

def center_xyzcoord(coordinates: list = []):
    if coordinates:
        for i, coord in enumerate(coordinates):
            coord[0] -= coordinates[-1][0]
            coord[1] -= coordinates[-1][1]
            coord[2] -= coordinates[-1][2]
    # centers a list in a format of [[x1, y1, z1],...,[xn, yn, zn]] around the nth coordinate
    return coordinates

def landmarklist_to_xyzcoord(landmark_list, all_52 : bool = True, centered: bool = False):
    # landmark_list must be a NormalizedLandmarkList Object
    if landmark_list:
        if all_52:
            landmark_coordiantes = [[landmark.x, landmark.y, landmark.z] for landmark in landmark_list.landmark] if len(landmark_list.landmark) == 52 else []
        else:
            landmark_coordiantes = [[landmark.x, landmark.y, landmark.z] for landmark in landmark_list.landmark]
    else:
        return []
    # converts a NormalizedLandmarkList with n NormalizedLandmarks into a list in the format of [[x1, y1, z1],...,[xn, yn, zn]] 
    return landmark_coordiantes if not centered else center_xyzcoord(landmark_coordiantes)

def center_xyzlandmarks(landmarks_list):
    if landmarks_list:
        for landmark in landmarks_list.landmark:
            landmark.x -= landmarks_list.landmark[-1].x
            landmark.y -= landmarks_list.landmark[-1].y
            landmark.z -= landmarks_list.landmark[-1].z
        
    return landmarks_list
    



