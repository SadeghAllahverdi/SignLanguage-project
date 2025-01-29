#Explanation: This python file contains functions for preprocessing our data.
#------------------------------------------------------------------------------Import--------------------------------------------------------------------------

# importing libraries for preprocessing data
import random 
import numpy as np
import torch
from collections import defaultdict
from sklearn.model_selection import train_test_split

#importing tqdm for progression bar and typing and numpy.typing for writing input types for each function
from tqdm.auto import tqdm 
from typing import List, Tuple
from numpy.typing import NDArray

#-----------------------------------------------------------------------Interpolation--------------------------------------------------------------------------
# function to interpolate 2 video detections
def interpolate_video_detections(video_detection_1: NDArray[np.float64], 
                                 video_detection_2: NDArray[np.float64], 
                                 frame_structure: List[Tuple[int, int]],
                                 alpha: float):
    """
    This function gets two video detection arrays and based interpolates them frame by frame. to make correct interpolations the function
    first checks , if both frames contain same body parts.
    Args:
        video_detection_1: First video detection array.
        video_detection_2: Second video detection array.
        frame_structure: represents the start and end index for each landmark class: pose, face, lh, rh.
        alpha: interpolation factor
    Returns:
        an array that is the interpolation of the two input video detections:
        inter_vid_detection
    Example usage: 
        inter_vid_detection = interpolate_video_detections(video_detection_1= v1, video_detection_2= v2, frame_structure= frame_structure, alpha= 0.5)
    """
    num_frames = video_detection_1.shape[0]                # number of frames that will be interpolated
    inter_vid_detection= np.zeros_like(video_detection_1)  # zero array for storing interpolated values
    for i in range(num_frames):
        frame_detection_1= video_detection_1[i]             
        frame_detection_2= video_detection_2[i]             
        inter_frame_detection= np.zeros_like(frame_detection_1) # stores interpolated frame
        
        for (start, end) in frame_structure:
            bodypart1= frame_detection_1[start:end]    # body part in frame
            bodypart2= frame_detection_2[start:end]    # body part in frame
    
            if np.all(bodypart1 == 0) and np.all(bodypart2 == 0):       # if the body part does not exist in both frames
                inter_frame_detection[start:end] = np.zeros(end- start) # put zero    

            elif np.all(bodypart1 == 0):                                # if body part 1 does not exist                   
                inter_frame_detection[start:end] = bodypart2            # put bodypart 2
            
            elif np.all(bodypart2 == 0):                                # if body part 2 does not exist                     
                inter_frame_detection[start:end] = bodypart1            # put bodypart 1
            
            else:  # if both exists then we interpolate
                inter_frame_detection[start:end]= (1 - alpha) * bodypart1 + alpha * bodypart2
                
        inter_vid_detection[i]= inter_frame_detection 
    return inter_vid_detection

# function to apply the interpolation to the entire dataset
def interpolate_dataset(detections: NDArray[np.float64],
                        labels: List[str],
                         alpha: float= 0.5,
                         noise_level: float= 0.001):
    """
    This function applies interpolation accross the entire dataset. It only interpolates between videos that have the same label. 
    Args:
        detections: array of all video detections from LSA64 or WLASL100 dataset
        labels: list of all video labels in the dataset
        alpha: interpolation factor.
        num_interpolations_samples: number of interpolated samples that should be produced for each label
    Returns:
        a tuple of (np.array(x), y) where np.array(x) is the detections and y is the labels
    Example usage:
        detections, labels = interpolate_dataset(detections, labels, alpha= 0.5, min_interpolations= 13)
    """
    current_data= defaultdict(list)                 # stores current data
    interpolated_data= defaultdict(list)            # stores interpolated data
    
    frame_structure= [(0, 132), (132, 1536), (1536, 1599), (1599, 1662)]  # represents the indexes of the concatenated pose, face, lh, rh
    
    x = []  #stores augmented detections
    y = []  #stores augmented labels

    # making a dictionary where key is label and value is list of all videos with same label
    for idx, label in enumerate(labels):
        current_data[label].append(detections[idx])

    # for each label, finding all video pair combinations:
    for label, video_detections in current_data.items():
        pairs= []
        for i in range(len(video_detections)):
            for j in range(i+1, len(video_detections)):
                pairs.append((i, j))
        # randomly select a number of pairs equal to the number of samples that are available for that label
        selected_pairs = random.sample(pairs, len(video_detections))
        # interpolating the randomly selected pairs
        for (i, j) in selected_pairs:
            video_detection_1= video_detections[i]
            video_detection_2= video_detections[j]
            
            inter_vid_detection = interpolate_video_detections(video_detection_1, video_detection_2, frame_structure, alpha) #interpolate
            # adding random gaussian noise
            noise = np.random.normal(0, noise_level, inter_vid_detection.shape[1:])  
            noisy_interpolated = np.clip(inter_vid_detection + noise, 0.001, 0.999)
            # adding the new sample under the label it belongs to
            interpolated_data[label].append(noisy_interpolated)
            
    # add video detections of both current and interpolated data together 
    for label in current_data:
        original_videos = current_data[label]  # Original samples
        interpolated_videos = interpolated_data[label]  # Interpolated samples

        combined_videos = original_videos + interpolated_videos
        sampled_videos = random.sample(combined_videos, len(original_videos))  # Randomly pick samples so that the original number of samples is preserved

        for video_detection in sampled_videos:
            x.append(video_detection)
            y.append(label)

    return np.array(x), y

#-------------------------------------------------------------------------Split Data--------------------------------------------------------------------------
#function to convert detections and labels to the right format for training
def convert(detections: NDArray[np.float64],
            labels: List[str],
            class_names: List[str]):
    """
    This function maps our Labels to numbers so that they are prepared for the training phase (ex: it maps the label "Red" to number 1). It also changes the
    detections from float64 to float32. since float64 would generate errors when training.
    Args:
        detections: array of all video detections
        label: labels for each video detection
        class_names: list of all class names withing the dataset. it is used to make a dictionray that converts labels to numbers.
    Returns:
        a tuple of (X, y) where X is our features/ detections and has type tensor float 32 and y is our label and has type long.
    Example use:
        X, y= convert(detections= detections, labels= labels, class_names= wlasl100class_names)
    """
    label_to_number= {label: num for num, label in enumerate(class_names)} # used for mapping the labels to numbers
    X= torch.tensor(detections, dtype=torch.float32)
    y= [label_to_number[label] for label in labels]                        # a list that has all the labels but in number format
    y= torch.tensor(y, dtype=torch.long)    
    
    return X, y

# fuction that splits the dataset for training
def split_dataset(detections: NDArray[np.float64],
                  labels: List[str],
                  class_names: List[str],
                  test_size: float):
    """
    This function splits the dataset and converts them so that they are suitable for training process. 
    Args:
        detections: video detections for the entire dataset.
        labels: list of all video labels for the entire dataset.
        class_names: list of all class names in the dataset
        test_size: determines how data should be splitted    
    Returns:
        a tuple of (X_train, X_test, y_train, y_test) 
    Example usage:
        xtrain, xtest, ytrain, ytest= split_dataset(detections, labels, class_names, 0.2)
    """
    X_train, X_test, y_train, y_test = train_test_split(detections, labels, test_size= test_size, random_state= 42, stratify=labels)
    X_train, y_train= convert(X_train, y_train, class_names)
    X_test, y_test= convert(X_test, y_test, class_names)
    
    return X_train, X_test, y_train, y_test
        