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
                A = fd1part+ ((fd1part + fd2part) / 2)**2 - (fd1part)**2 + ((fd1part + fd2part) / 2)**2 - (fd2part)**2 # Interpolate normally
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



def convert(detections: NDArray[np.float64],
            labels: List[str],
            class_names: List[str]):
    """
    changes detection from float 64 to float 32. and maps labels to numbers using a dictionary 
    """
    label_map= {label: num for num, label in enumerate(class_names)}
    X= torch.tensor(detections, dtype=torch.float32)
    y= [label_map[label] for label in labels] 
    y= torch.tensor(y, dtype=torch.long)    
    
    return X, y


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
    
    X_train, X_test, y_train, y_test = train_test_split(detections, labels, test_size=0.2, random_state=42, stratify=labels)
    X_train, y_train= convert(X_train, y_train, class_names)
    X_test, y_test= convert(X_test, y_test, class_names)
    
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
