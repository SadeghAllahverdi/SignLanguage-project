# importing libraries for preprocessing data
import os                                                                           
import random 
import json
import numpy as np                                                                  
import pandas as pd
import matplotlib.pyplot as plt                                                       

import cv2                                                                         
import mediapipe as mp  

import torch
from torch.utils.data import Dataset
from glob import glob                                                               
from pathlib import Path                                                            
from natsort import natsorted                                                       
from tqdm.auto import tqdm                                                         
from collections import defaultdict
from typing import Callable, List, Tuple
from sklearn.model_selection import train_test_split
from numpy.typing import NDArray

# functions to implement interpolation 
def interpolate_video_detections(vd1: NDArray[np.float64], 
                                 vd2: NDArray[np.float64], 
                                 shapes: List[Tuple[int, int]],
                                 alpha: float):
    """
    This function will return an Array which is in principle the interpolation of 2 video detection arrays.
    Args:
        vd1: landmarks detected in first video.
        vd2: landmarks detected in second video.
        shapes: represents the start and end index for each landmark class: head , pose, left hand and right hand.
        alpha: interpolation factor
        
    Returns:
        either: (1 - alpha) * most_recent_detection + alpha * next_coming_detection
        or: next_coming_detection
        or: most_recent_detection

    Example usage: 
        
    """
    num_frames = vd1.shape[0]
    ivd= np.zeros_like(vd1)
    for i in range(num_frames):
        fd1= vd1[i]    # frame in video 1
        fd2= vd2[i]    # frame in video 2
        ifd= np.zeros_like(fd1)  # stores interpolated frame
        for (start, end) in shapes:         # goes through each landmark class and interpolates
            fd1part= fd1[start:end]
            fd2part= fd2[start:end]
            if np.all(fd1part == 0) and np.all(fd2part == 0): # all pose or face or lh or rh set of landmarks are 0 or not detected
                ivd[i][start:end] = np.zeros(end- start)      # put zero
            elif np.all(fd1part == 0):                        # all pose or face or lh or rh only in frame of video 1 is are not detected
                ivd[i][start:end] = fd2part                   # use video 2 pose or face or lh or rh
            elif np.all(fd2part == 0):                        # all pose or face or lh or rh only in frame of video 2 is are not detected
                ivd[i][start:end] = fd1part                   # use video 1 pose or face or lh or rh
            else:
                #ivd[i][start:end]= (1 - alpha) * fd1part + alpha * fd2part
                # this formula also works very nice
                A = fd1part+ ((fd1part + fd2part) / 2)**2 - (fd1part)**2 + ((fd1part + fd2part) / 2)**2 - (fd2part)**2  # Interpolate normally
                B = fd1part+ ((fd1part + fd2part) / 2)**2 - (fd1part)**2 + ((fd1part + fd2part) / 2)**2 - (fd2part)**2 
                ivd[i][start:end]= (1 - alpha) * A + alpha * B
    return ivd


def interpolate_dataset(detections: NDArray[np.float64],
                        labels: List[str],
                        alpha: float= 0.5,
                        min_interpolations: int= 5):
    """
    This function does interpolation on videos with same label across the dataset. 
    Args:
        detections: landmark dataset array.
        labels: list of labels.
        alpha: interpolation factor.
        min_interpolations: minimum number of interpolated videos produced for each label
        
    Returns:
        a tuple of (np.array(x), y) where np.array(x) is the landmark dataset array and
        y is the corresponding label for each video detection in the landmark dataset array

    Example usage:
        ivd = interpolate_video_detections(vid1, vid2, [(0, 132), (132, 1536), (1536, 1599), (1599, 1662)],0.5, 5) 
    """
    data= defaultdict(list)                         # stores current data
    interpolated_data= defaultdict(list)            # stores interpolated data
    augumented_data = defaultdict(list)             # union of current and interpolated
    detection_shape= [(0, 132), (132, 1536), (1536, 1599), (1599, 1662)]  # represents how face, pose, lh, rh are stored
    
    x = []                                          # used to unpack augumented_data
    y = []

    for idx, label in enumerate(labels):
        data[label].append(detections[idx])

    for label, detections in data.items():
        pairs= []
        for i in range(len(detections)):
            for j in range(i+1, len(detections)):
                pairs.append((i, j))
        selected_pairs = random.sample(pairs, min(min_interpolations, len(pairs)))
        for (i, j) in selected_pairs:
            ivd = interpolate_video_detections(detections[i], detections[j], detection_shape,alpha)
            interpolated_data[label].append(ivd)

    for d in (data, interpolated_data):
        for label, detections in d.items():
            augumented_data[label].extend(detections)

    for label, detections in augumented_data.items():
        for detection in detections:
            x.append(detection)
            y.append(label)

    return np.array(x), y


# functions for drawing video landmarks
def draw_landmarks(frame: np.ndarray,
                   detection: NDArray[np.float64],
                   structure: List[int]):
    """
    This function traverses in the video array and draws the detections frame by frame.  
    Args:
        frame: represents frame that is shown
        detection: array that represents video frame by frame.
        structure list of integers where each entary says how many dimention does this landmark has: 
        
    Returns:
        manipulated frame 

    Example usage:
        frame = draw_landmarks(frame, vid, [4]*33 + [3]*(468 + 21 + 21))
    """
    idx = 0
    while idx < len(detection):
        for coordinates in structure:
            if idx + coordinates > len(detection): # makes sure the index stays in the video
                break  

            x, y = detection[idx], detection[idx + 1] # get x, y values
            # multiply x and y to hight and width of the frame
            px = int(x * frame.shape[1]) 
            py = int(y * frame.shape[0])
            cv2.circle(frame, (px, py), 3, (0, 255, 0), -1) # draws the landmark
            
            idx += coordinates # base on value of coordinate (either 3 or 4)jumps to next landmark in principle
            if idx >= len(detection): 
                break
    return frame


def show_detections(vd: NDArray[np.float64]):
    """
    This function draws x and y landmarks of a video .  
    Args:
        vd: an array that represents video detections 
    """
    structure= [4]*33 + [3]*(468 + 21 + 21)  
    height= 720
    width= 1280
    cv2.namedWindow("Landmark Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Landmark Preview", width= width, height= height)
    try:
        for idx, detection in enumerate(vd):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame = draw_landmarks(frame, detection, structure)
            cv2.imshow("Landmark Preview", frame)
            if cv2.waitKey(100) & 0xFF == 27:  #ESC key
                break
            #print(f"Displaying frame {idx + 1}/{len(vd)}")
    finally:
        cv2.destroyAllWindows()


def split_dataset(detections: NDArray[np.float64],
                  labels: List[str],
                  class_names: List[str],
                  test_size: float,
                  random_state: int):
    """
    splits the dataset and maps each video label to a corresponding number that is suitable for training process.  
    Args:
        detections: mediapipe detections for entire dataset.
        labels: list of all video labels for the entire dataset.
        class_names: list of all class names in the dataset
        test_size: determines how data should be splitted
        random_state:
        
    Returns:
        a tuple of (X_train, X_test, y_train, y_test) 

    Example usage:
        xtrain, xtest, ytrain, ytest= split_dataset(detections, labels, class_names, 0.2, 42)
    """
    label_map= {label: num for num, label in enumerate(class_names)}
    
    X_train, X_test, y_train, y_test = train_test_split(detections, labels, test_size=0.2, random_state=42, stratify=labels)
    X_train, X_test= torch.tensor(X_train, dtype=torch.float32) , torch.tensor(X_test, dtype=torch.float32)
    
    y_train= [label_map[label] for label in y_train]
    y_test= [label_map[label] for label in y_test]
    y_train= torch.tensor(y_train, dtype=torch.long)
    y_test= torch.tensor(y_test, dtype=torch.long)
    
    return X_train, X_test, y_train, y_test


# dataset class
class CustomDataset(Dataset):
    def __init__(self,features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        return feature, label
