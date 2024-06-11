# Explanation: This file will contain functions that are used to extract landmarks and make data sets
# importing libraries for making datasets

import os
import json
import random 

import numpy as np                                                                  
import pandas as pd
import matplotlib.pyplot as plt                                                       

import cv2                                                                         
import mediapipe as mp  

from glob import glob                                                               
from pathlib import Path                                                            
from natsort import natsorted                                                       
from tqdm.auto import tqdm                                                         
from typing import Callable, List

# function to get landmarks from LSA64 dataset.
def get_landmarks_LSA64(root: str,
                        class_names: List[str],                    
                        frame_numbers: int):
    """
    Goes through the LSA64_directory and creates a list of landmarks for each video within the directory by applying mediapipe model frame by frame.
    Args:
        root: Path to video dataset directory.
        class_names: List of all words in the dataset.
        frame_numbers: number of frames we want to take from the entire video: this is in principle fps.
        
    Returns:
        A tuple of (detections, labels, len(all_video_paths),len(none_cv2_video_paths)).
        Where detections is a list of all videos as mediapipe landmarks.
        labels is a list of labels corresponding to each video detection.
        len(all_video_paths),len(none_cv2_video_paths) are returned to see if cv2 was unable to open some
        video files.

    Note:
        video_detections has the following structure:
        indexes (0 to 131) of the list correspond to the first 33 pose landmarks : [x, y, z, visibility]
        indexes (132 to 1535) of the list correspond to the first 468 face landmarks: [x, y, z]
        indexes (1536 to 1598) of the list correspond to the first 21 left hand landmarks: [x, y, z]
        indexes (1599 to 1661) of the list correspond to the first 21 right hand landmarks: [x, y, z]

        in total 1662 coordinate/visibility values OR 543 total landmark objects
    Example use:
        results= get_landmarks_LSA64(root, lsa64class_names, 30)
    """
    all_video_paths= natsorted([str(p) for p in Path(root).glob("**/*.mp4")])      # makes list of video paths in the dataset
    vid_idx_to_class_name= {i+1:label for i, label in enumerate(class_names)}      # this is used to change the video numbers to the class name since videos are encoded in a specific way
    none_cv2_video_paths= []                                                       # stores video paths that cv2 couldnt open
    detections= []                                                                 # stores detections
    labels= []                                                                     # stores video labels
    frame_numbers= frame_numbers
    with mp.solutions.holistic.Holistic(min_detection_confidence= 0.5, min_tracking_confidence=0.5) as holistic:
        for video_path in tqdm(all_video_paths, desc="Processing videos"):
            cap = cv2.VideoCapture(video_path)              # Read each video using cv2
            if not cap.isOpened():                          # if cv2 can't read the video
                none_cv2_video_paths.append(video_path)     # save the video path
            else:                                           # if cap can read the video
                total_frames_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))                                # getting the total frames in video
                frame_idxs_to_process = np.linspace(0, total_frames_number-1, frame_numbers, dtype=int)     # picking desiered frame indexes
                video_detections= []
                for idx in frame_idxs_to_process:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame= cap.read()
                    #
                    result= holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    pose= np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark]).flatten() if result.pose_landmarks else np.zeros(33*4) 
                    face= np.array([[res.x, res.y, res.z] for res in result.face_landmarks.landmark]).flatten() if result.face_landmarks else np.zeros(468*3) 
                    lh= np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]).flatten() if result.left_hand_landmarks else np.zeros(21*3)
                    rh= np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten() if result.right_hand_landmarks else np.zeros(21*3)
                    detection= np.concatenate((pose,face,lh, rh))
                    video_detections.append(detection)

                
                detections.append(video_detections)    
                label= vid_idx_to_class_name[int(os.path.basename(video_path).split('_')[0])] # gets the label for the videos
                labels.append(label)
   
            cap.release()
        
    return detections, labels, len(all_video_paths),len(none_cv2_video_paths)

# functions to get landmarks from WLASL100 dataset.

def process_frame(cap, index):
    """
    Positions the cv2 on a specific frame number, reads the frame and returns the results.
    Example use:
        ret, frame= process_frame(cap, idx)
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    return cap.read() 


def get_landmarks_WLASL100(root: str,
                           class_names: List[str],
                           frame_numbers: int = 25):
    """
    Goes through the WLASL100 directory and creates a list of landmarks for each video within the directory by applying mediapipe model frame by frame.
    since some of the videos have faulty frames. it checks for before and after frames first. incase those are faulty as well it puts an empty list
    for that frame of the video.
    Args:
        root: Path to video dataset directory.
        class_names: List of all words in the dataset.
        frame_numbers: number of frames we want to take from the entire video: this is in principle fps.
        
    Returns:
        A tuple of (detections, labels, len(all_video_paths),len(none_cv2_video_paths)).
        Where detections is a list of all videos as mediapipe landmarks.
        labels is a list of labels corresponding to each video detection.
        len(all_video_paths),len(none_cv2_video_paths) are returned to see if cv2 was unable to open some
        video files.

    Note:
        video_detections has the following structure:
        indexes (0 to 131) of the list correspond to the first 33 pose landmarks : [x, y, z, visibility]
        indexes (132 to 1535) of the list correspond to the first 468 face landmarks: [x, y, z]
        indexes (1536 to 1598) of the list correspond to the first 21 left hand landmarks: [x, y, z]
        indexes (1599 to 1661) of the list correspond to the first 21 right hand landmarks: [x, y, z]

        in total 1662 coordinate/visibility values OR 543 total landmark objects

    Example use:
        results= get_landmarks_WLASL100(root= root, 
                                        class_names= class_names,
                                        frame_numbers= 60):
    """
    all_video_paths= natsorted([str(p) for p in Path(root).glob("**/*.mp4")])
    vid_idx_to_class_name= {i+1:label for i, label in enumerate(class_names)}
    none_cv2_video_paths= []                                # keeping track of corrupt video files
    detections= []                                          # saves mediapipe detections
    labels= []
    
    with mp.solutions.holistic.Holistic(min_detection_confidence= 0.5, min_tracking_confidence=0.5) as holistic:
        for video_path in tqdm(all_video_paths, desc="Processing videos"):
            cap = cv2.VideoCapture(video_path)              # Read each video using cv2
            if not cap.isOpened():                          # if cv2 can't read the video
                none_cv2_video_paths.append(video_path)     # save the video path
                continue
                
            total_frames_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))                                             # getting the total frames in video
            frame_idxs_to_process = np.round(np.linspace(0, total_frames_number - 1, frame_numbers)).astype(int)     # picking desiered frame indexes
            video_detections= []
            for idx in frame_idxs_to_process:
                ret, frame= process_frame(cap, idx)

                if ret is False:
                    # if the return value is False: meaning the frame was "bad".
                    print(f"Failed to grab frame {idx}, of video {video_path} of length {total_frames_number} frames. trying adjacent frames...")

                    # we try to read the previous frame.
                    ret, frame = grab_frame(cap, idx - 1)
                    if not ret:
                        # if the return value is False: meaning previous frame was also "bad".
                        # we try to read the next frame with cv2.
                        ret, frame = grab_frame(cap, idx + 1)
                        
                if not ret:
                    # if the return value is False.
                    print(f"Unable to retrieve any frames around index {idx}, of video {video_path} of length {total_frames_number} frames.")
                    # we add empty detection that will be filled using interpolation
                    detection= []
                    video_detections.append(detection)
                    continue
                            
                result= holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pose= np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark]).flatten() if result.pose_landmarks else np.zeros(33*4) 
                face= np.array([[res.x, res.y, res.z] for res in result.face_landmarks.landmark]).flatten() if result.face_landmarks else np.zeros(468*3) 
                lh= np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]).flatten() if result.left_hand_landmarks else np.zeros(21*3)
                rh= np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten() if result.right_hand_landmarks else np.zeros(21*3)
                detection= np.concatenate((pose,face,lh, rh))
                video_detections.append(detection)

                
            detections.append(video_detections)    
            label= vid_idx_to_class_name[int(os.path.basename(os.path.dirname(video_path)))]
            labels.append(label)
   
        cap.release()
        
    return detections, labels, len(all_video_paths),len(none_cv2_video_paths)


# functions to interpolate faulty video frames in the dataset.
def interpolate_frames(most_recent_detection, next_coming_detection, alpha):
    """
    Based on the value of most recent detection and next coming detection which are the frames before and after our faulty frame returns 
    a landmark array for the faulty frame.
    Args:
        most_recent_detection: landmarks detected in previous frame.
        next_coming_detection: landmarks detected in the next frame.
        alpha: interpolation factor.
        
    Returns:
        either: (1 - alpha) * most_recent_detection + alpha * next_coming_detection
        or: next_coming_detection
        or: most_recent_detection

    Example use:
        video_detection[i]= interpolate_frames(most_recent_detection, next_coming_detection, 0.5)
    """
    if most_recent_detection is None and next_coming_detection is not None:             # first to nth frames are all corrupt
        return next_coming_detection
    elif most_recent_detection is not None and next_coming_detection is None:           # nth to last frames are all corrupt
        return most_recent_detection
    else:
        return (1 - alpha) * most_recent_detection + alpha * next_coming_detection 


def fill_empty_detections(result):
    """
    In principle fills up the empty landmark detections for frames that where faulty in the dataset and returns the dataset .
    Args:
        detections
        
    Returns:
        detections (with no empty landmark frame)

    Example use: detections= fill_empty_detections(detections)
    """
    detections= result
    for video_detection in detections:
        most_recent_detection= None
        for i in range(len(video_detection)):
            if len(video_detection[i]) != 0:
                most_recent_detection= video_detection[i]
            else:
                next_coming_detection= None
                for j in range(i+1, len(video_detection)):
                    if len(video_detection[j]) != 0:
                        next_coming_detection= video_detection[j]
                        break
                    else:
                        continue
                     
                video_detection[i]= interpolate_frames(most_recent_detection, next_coming_detection, 0.5)
                most_recent_detection= video_detection[i]

    return detections

wlasl100class_names= ["accident", "africa", "all", "apple", "basketball", "bed", "before", "bird", "birthday",
                      "black", "blue", "bowling", "brown", "but", "can", "candy", "chair", "change", "cheat", "city",
                      "clothes", "color", "computer", "cook", "cool", "corn", "cousin", "cow", "dance", "dark",
                      "deaf", "decide", "doctor", "dog", "drink", "eat", "enjoy", "family", "fine", "finish",
                      "fish", "forget", "full", "give", "go", "graduate", "hat", "hearing", "help", "hot",
                      "how", "jacket", "kiss", "language", "last", "letter", "like", "man", "many", "meet",
                      "mother", "need", "no", "now", "orange", "paint", "paper", "pink", "pizza", "play",
                      "pull", "purple", "right", "same", "school", "secretary", "shirt", "short", "son", "study",
                      "table", "tall", "tell", "thanksgiving", "thin", "thursday", "time", "walk", "want", "what",
                      "white", "who", "woman", "work", "wrong", "year", "yes", "book", "later", "medicine"]


lsa64class_names= ['Opaque', 'Red', 'Green', 'Yellow', 'Bright', 'Light-blue', 'Colors', 'Pink',
                   'Women', 'Enemy', 'Son', 'Man', 'Away', 'Drawer', 'Born', 'Learn',
                   'Call', 'Skimmer', 'Bitter', 'Sweet milk', 'Milk', 'Water', 'Food', 'Argentina',
                   'Uruguay', 'Country', 'Last name', 'Where', 'Mock', 'Birthday', 'Breakfast', 'Photo',
                   'Hungry', 'Map', 'Coin', 'Music', 'Ship', 'None', 'Name', 'Patience',
                   'Perfume', 'Deaf', 'Trap', 'Rice', 'Barbecue', 'Candy', 'Chewing-gum', 'Spaghetti',
                   'Yogurt', 'Accept', 'Thanks', 'Shut down', 'Appear', 'To land', 'Catch', 'Help',
                   'Dance', 'Bathe', 'Buy', 'Copy', 'Run', 'Realize', 'Give', 'Find']
