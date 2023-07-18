import mediapipe as mp
import matplotlib.pyplot as plt
import queue
from data_processing import *
from data_collection import *
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

import queue
import pandas as pd

def csv_to_landmark_frames(path: str = None, start_row: int = 3, num_columns: int = 52):
    # only takes csv that are created by create_csv() from data_collection.py
    df = pd.read_csv(path, index_col = 0, skiprows = range(1, start_row))
    landmark_frames = queue.Queue()
    for i, column in enumerate(df):
        df[column] = df[column].astype(float)
    for i  in range(1, len(df) + 1):
        container = []
        for j in range(int(len(df.columns)/3)):
            if j < num_columns:
                container += [[df[str(j)][i], df[str(j+0.1)][i], df[str(j+0.2)][i]]]
            else:
                break
        landmark_frames.put(container)

    # Returns a queue of lists with a format of [[x1, y1, z1],...,[xn, yn, zn]]
    return landmark_frames



def get_frame(frames = queue.Queue()):
    return frames.get() if not frames.empty() else []

def animate_landmark_frames(empty = None, frames = queue.Queue(), elevation: int = 10, azimuth: int = 10):
   

    def animate_landmarks(empty = None, frames = frames, 
                          elevation: int = elevation, azimuth: int = azimuth, 
                          index: tuple = (47, 51)):

        # This is a modified version of the built-in function mediapipe.solutions.drawing_utils.plot_landmarks
        # The original function can be found on path\mediapipe\python\solutions\drawing_utils.py
        # The modifications in this code was done so that it will be able to dispaly a live feed of the landmarks
        included_landmarks = [i for i in range(index[0], index[1] + 1)]
        landmarks = center_xyzcoord(get_frame(frames))
        # landmark_list = _landmarks_list
        connections0 = get_connection_list()
        if not landmarks:
            return

        plt.cla()
        ax.view_init(elev=elevation, azim=azimuth)
        plotted_landmarks = {}
        for idx, landmark in enumerate(landmarks):
            if idx not in included_landmarks:
                continue
            ax.scatter3D(
                xs=[-landmark[2]],
                ys=[landmark[0]],
                zs=[-landmark[1]],
                color='r')
            plotted_landmarks[idx] = (-landmark[2], landmark[0], -landmark[1])
        if connections0:
            num_landmarks = len(landmarks)
            # Draws the connections if the start and end landmarks are both visible.
            for connection in connections0:
                start_idx = connection[0]
                end_idx = connection[1]
                if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                    raise ValueError(f'Landmark index is out of range. Invalid connection '
                                    f'from landmark #{start_idx} to landmark #{end_idx}.')
                if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                    landmark_pair = [
                        plotted_landmarks[start_idx], plotted_landmarks[end_idx]
                    ]
                    ax.plot3D(
                        xs=[landmark_pair[0][0], landmark_pair[1][0]],
                        ys=[landmark_pair[0][1], landmark_pair[1][1]],
                        zs=[landmark_pair[0][2], landmark_pair[1][2]],
                        color='g')
        lim = 0.5
        ax.set_xlim(-lim, lim)   # Set x-axis limits
        ax.set_ylim(-lim, lim)   # Set y-axis limits
        ax.set_zlim(-lim, lim)   # Set z-axis limits

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    animate = FuncAnimation(plt.gcf(), animate_landmarks, interval=20)
    plt.show()

# path = 'data\data-2023-07-15-131838.csv'
# frames = csv_to_landmark_frames(path=path)
# animate_landmark_frames(frames=frames)


