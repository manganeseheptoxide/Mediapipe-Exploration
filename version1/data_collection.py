import queue
import pandas as pd
from datetime import datetime
from data_processing import *
from mediapipe.framework.formats import landmark_pb2

NormLandmark = mp.tasks.components.containers.NormalizedLandmark
NormLandmarkList = landmark_pb2.NormalizedLandmarkList

def create_csv(df: pd.DataFrame = pd.DataFrame(), index: bool = True):
    filename = str('data/data-'+datetime.now().strftime("%Y-%m-%d-%H%M%S")+'.csv')
    df.to_csv(filename, index = index)
    print('Created csv file.')

def create_xyz_landmark_df(num_landmarks: int = 52, num_frames: int = 0):
    indexes = []
    for i in range(num_landmarks):
        indexes += [(i, axis) for axis in ['x', 'y', 'z']]
    columns = pd.MultiIndex.from_tuples(indexes, names = ['Landmarks', 'Coordinates'])
    df_3d = pd.DataFrame(index=[frames + 1 for frames in range(num_frames)], columns = columns)
    df_3d.index.name = 'Frame'
    
    return df_3d

def add_df_entry(df: pd.DataFrame = pd.DataFrame(), entry: list = []):

    if entry:

        if not len(df.columns) == len(entry)*3:
            raise ValueError('The number of landmarks in the dataframe and entry is not equal')

        if df.empty:
            df.loc[1] = None
            for i in range(len(entry)):
                df.loc[1][i].x = entry[i][0]
                df.loc[1][i].y = entry[i][1]
                df.loc[1][i].z = entry[i][2]
        else:
            frame = len(df) + 1
            df.loc[frame] = None
            for i in range(len(entry)):
                df.loc[frame][i].x = entry[i][0]
                df.loc[frame][i].y = entry[i][1]
                df.loc[frame][i].z = entry[i][2]

    return df

def df_entry_from_queue_List(df: pd.DataFrame = pd.DataFrame(),
                             data_cache = queue.Queue(),
                             status: bool = True,
                             centered: bool = False):
    
    while not data_cache.empty():
        data = data_cache.get()
        if centered:
            data = center_xyzcoord(data)
        df = add_df_entry(df = df, entry = data)
        if status:
            print('Data added')
    if status:
        print('No queued landmark data')
    return df

def df_entry_from_queue_NLL(df: pd.DataFrame = pd.DataFrame(),
                            data_cache = queue.Queue(),
                            status: bool = True,
                            centered: bool = False):
    
    while not data_cache.empty():
        data = landmarklist_to_xyzcoord(data_cache.get(), centered = centered)
        df = add_df_entry(df = df, entry = data)
        if status:
            print('Data added')
    if status:
        print('No queued landmark data')
    return df


