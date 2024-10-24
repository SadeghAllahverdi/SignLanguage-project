# Explanation: This python file contains functions that are used to extract landmarks from LSA64, AUTSL40 and WLASL100 datasets.

#------------------------------------------------------------------------------Import--------------------------------------------------------------------------

# importing libraries for working with directories of the libreries
import os
from pathlib import Path
from natsort import natsorted 
# importing OpenCV and Mediapipe to read videos and extract landmarks
import cv2                                                                         
import mediapipe as mp  
# importing numpy to work with arrays
import numpy as np                                                                                                                           
# importing tqdm for progression bar and typing for writing input types for each function                                                      
from tqdm.auto import tqdm                                                         
from typing import Callable, List

#------------------------------------------------------------------constant variables--------------------------------------------------------------------------
# A list for all class names in AUTSL 40
autslclass_names = ["sister", "hurry", "hungry", "enjoy_your_meal", "brother", "tree", "heavy", "cry", "family", "wise", "unwise", "kin", "shopping", "key",
                    "mother", "friend", "ataturk", "shoe", "mirror", "same", "father", "garden", "look", "honey", "glass", "flag", "feast", "baby", "single",
                    "wait", "I", "petrol", "together", "inform", "we", "work", "wednesday", "fork", "tea", "teapot"]

# A list for all class names in LSA64
lsa64class_names= ['Opaque', 'Red', 'Green', 'Yellow', 'Bright', 'Light-blue', 'Colors', 'Pink', 'Women', 'Enemy', 'Son', 'Man', 'Away', 'Drawer', 'Born',
                   'Learn', 'Call', 'Skimmer', 'Bitter', 'Sweet milk', 'Milk', 'Water', 'Food', 'Argentina', 'Uruguay', 'Country', 'Last name', 'Where',
                   'Mock', 'Birthday', 'Breakfast', 'Photo', 'Hungry', 'Map', 'Coin', 'Music', 'Ship', 'None', 'Name', 'Patience','Perfume', 'Deaf', 'Trap',
                   'Rice', 'Barbecue', 'Candy', 'Chewing-gum', 'Spaghetti', 'Yogurt', 'Accept', 'Thanks', 'Shut down', 'Appear', 'To land', 'Catch', 'Help',
                   'Dance', 'Bathe', 'Buy', 'Copy', 'Run', 'Realize', 'Give', 'Find']

# A list for all class names in WLALS100
wlasl100class_names = ["accident", "africa", "all", "apple", "basketball", "bed", "before", "bird", "birthday", "black", "blue", "book", "bowling", "brown",
                       "but", "can", "candy", "chair", "change", "cheat", "city", "clothes", "color", "computer", "cook", "cool", "corn", "cousin", "cow",
                       "dance", "dark", "deaf", "decide", "doctor", "dog", "drink","eat", "enjoy", "family", "fine", "finish", "fish", "forget", "full",
                       "give", "go", "graduate", "hat", "hearing", "help", "hot", "how", "jacket", "kiss", "language", "last", "later", "letter", "like",
                       "man", "many", "medicine", "meet", "mother", "need", "no", "now", "orange", "paint", "paper", "pink", "pizza", "play", "pull", "purple",
                       "right","same", "school", "secretary", "shirt", "short", "son", "study", "table", "tall", "tell", "thanksgiving", "thin", "thursday",
                       "time", "walk", "want", "what", "white", "who", "woman", "work", "wrong", "year", "yes"]

#--------------------------------------------------------------------Getting landmarks--------------------------------------------------------------------------
# function to get landmarks from LSA64 or AUTSL40 dataset.
def get_landmarks(root: str,
                  class_names: List[str],
                  frame_numbers: int):
    """
    This function retrieves all video paths from the dataset directory. Then the function analysis videos frame by frame and extract landmark. Finally the
    function is able to assigne each video, an array of detected landmarks. depending on the datas, the function also uses the title of each video to assign
    labels to them by using a dictionary.
    Args:
        root: Path to where dataset is located.
        class_names: List of all words in the dataset.
        frame_numbers: number of frames we want to take from the each video in the dataset.
    Returns:
        detections, labels, len(all_video_paths),len(none_cv2_video_paths) where:
        detections: is a list of all mediapipe landmarks that were detected from all videos.
        labels: is a list of labels corresponding to each video detection.
        len(all_video_paths): is the number of videos in the dataset.
        none_cv2_video_paths" is a list of videos that OpenCV was not able to open.
    Example use:
        results= get_landmarks_LSA64(root= root, class_names= lsa64class_names, frame_numbers= 30)
        detections, labels, num_all_videos, none_cv2_video_paths= results[0], results[1], results[2], results[3]
    """
    labels= []                       # a list to store video labels
    detections= []                   # a list to store all video detections
    none_cv2_video_paths= []         # a list to store video paths that cv2 can't capture
    
    all_video_paths= Path(root).glob("**/*.mp4")                           # a list to store all video paths in the dataset
    all_video_paths= [str(path) for path in all_video_paths]               # changing path objects to strings since natosrt works with strings
    all_video_paths= natsorted(all_video_paths)                            # sorted
    vid_idx_to_label= {i:label for i, label in enumerate(class_names)}     # this mapping is used to change the video titles to labels
    
    with mp.solutions.holistic.Holistic(min_detection_confidence= 0.5, min_tracking_confidence=0.5) as holistic:
        for video_path in tqdm(all_video_paths, desc="Processing videos"):
            cap = cv2.VideoCapture(video_path)                             # capture each video using OpenCV
            if not cap.isOpened():                                         # if OpenCV can't capture the video path
                none_cv2_video_paths.append(video_path)                    # add the video path to none_cv2_video_paths
            else:                                                                                                      
                video_detections= []                                                                     # a list to store video detections
                total_frames_number= cap.get(cv2.CAP_PROP_FRAME_COUNT)                                   # getting total number of frames from a video
                total_frames_number = int(total_frames_number)                                           # changing float to integer   
                frame_idxs_to_process = np.linspace(0, total_frames_number-1, frame_numbers, dtype=int)  # picking desiered frame indexes
                
                for idx in frame_idxs_to_process:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)                             # set the video to the desired frame index
                    ret, frame= cap.read()                                            # reading the frame 
                    result= holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # processing the frame (Mediapipe works with RGB)
                    pose,face,lh, rh= get_frame_detections(result)                    # turning results into flattened arrays
                    frame_detection= np.concatenate((pose,face,lh, rh))  
                    video_detections.append(frame_detection)                          # storing the frame detection in the video detection list
                    
                if class_names== autslclass_names:    # for AUTSL
                    video_idx= int(os.path.basename(os.path.dirname(video_path))) # extract video index from the video folder
                    label= vid_idx_to_label[video_idx]                            # map the video index to a label
                    
                elif class_names== lsa64class_names:  # for LSA64
                    video_idx= int(os.path.basename(video_path).split('_')[0])    # extract index from the video title: 001_004_003 -> 1
                    label= vid_idx_to_label[video_idx-1]                          # map the index to the correct label
                
                labels.append(label)
                detections.append(video_detections) 
   
            cap.release()
        
    return detections, labels, len(all_video_paths), none_cv2_video_paths

# function to get landmarks from WLASL100 dataset.
def get_landmarks_WLASL100(root: str,
                           class_names: List[str],
                           frame_numbers: int):
    """
    This function retrieves all video paths from the WLSA100 directory. Then the function analysis videos frame by frame and extract landmark. Since some of 
    the videos have faulty frames. it checks for before and after frames first. incase those are faulty as well it puts an empty list for that frame of the
    video. Finally the function is able to assigne each video, an array of detected landmarks and a label.    
    Args:
        root: Path to video WLASL100 dataset directory.
        class_names: List of all words in the dataset.
        frame_numbers: number of frames we want to take from the entire video.
   Returns:
        detections, labels, len(all_video_paths),len(none_cv2_video_paths) where:
        detections: is a list of all mediapipe landmarks that were detected from all videos.
        labels: is a list of labels corresponding to each video detection.
        len(all_video_paths): is the number of videos in the dataset.
        none_cv2_video_paths" is a list of videos that OpenCV was not able to open.       
    Example use:
        results= get_landmarks_WLASL100(root= root, class_names= class_names frame_numbers= 30)
        detections, labels, num_all_videos, none_cv2_video_paths= results[0], results[1], results[2], results[3]
    """
    labels= []                    # a list to store video labels
    detections= []                # a list to store all video detections
    none_cv2_video_paths= []      # a list to store video paths that cv2 can't capture
    
    all_video_paths= Path(root).glob("**/*.mp4")                        # a list to store all video paths in the dataset
    all_video_paths= [str(path) for path in all_video_paths]            # changing path objects to strings since natosrt works with strings
    all_video_paths= natsorted(all_video_paths)                         # sorted
    vid_idx_to_label= {i:label for i, label in enumerate(class_names)}  # this mapping is used to change the video titles to labels
    
    with mp.solutions.holistic.Holistic(min_detection_confidence= 0.5, min_tracking_confidence=0.5) as holistic:
        for video_path in tqdm(all_video_paths, desc="Processing videos"):
            cap = cv2.VideoCapture(video_path)              # capture each video using Opencv
            if not cap.isOpened():                          # if OpenCV can't capture the video
                none_cv2_video_paths.append(video_path)     # add the video path to none_cv2_video_paths list
            else:
                video_detections= []
                total_frames_number= cap.get(cv2.CAP_PROP_FRAME_COUNT)                                     # getting total number of frames from a video
                total_frames_number = int(total_frames_number)                                             # changing float to integer   
                frame_idxs_to_process = np.linspace(0, total_frames_number - 1, frame_numbers, dtype= int) # picking desiered frame indexes
                
                for idx in frame_idxs_to_process:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx) # set the video to the desired frame index
                    ret, frame= cap.read()                # read the frame
                    if not ret:                           # if the frame was "unreadable".
                        print(f"Failed to grab frame {idx}, of video {video_path} of length {total_frames_number} frames. trying adjacent frames...")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)      # set video to previous frame
                        ret, frame = cap.read()                        # read the frame
                        if not ret:                                    # if previous was also "unreadable"
                            cap.set(cv2.CAP_PROP_POS_FRAMES, idx + 1)  # set the video to next frame
                            ret, frame = cap.read()                    # read frame
                            
                    if not ret:                           # if the return value is still False
                        print(f"Unable to retrieve any frames around index {idx}, of video {video_path} of length {total_frames_number} frames.")
                        frame_detection= []               # we add empty detection that will be filled later, using interpolation
                        video_detections.append(frame_detection)
                        continue
                                
                    result= holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    pose,face,lh, rh= get_frame_detections(result)       # turning results into flattened arrays
                    frame_detection= np.concatenate((pose,face,lh, rh))   
                    video_detections.append(frame_detection)             # storing the frame detection in the video detection list

                video_idx= int(os.path.basename(os.path.dirname(video_path))) # extract video index from the folder
                label= vid_idx_to_label[video_idx-1]                          # map the video index to a label
                detections.append(video_detections)    
                labels.append(label)
       
            cap.release()
            
        return detections, labels, len(all_video_paths), none_cv2_video_paths

#----------------------------------------------------------------Additional functions--------------------------------------------------------------------------
def get_frame_detections(result):
    '''
    This function turns the result objects obtianed with mediapipe into flattened numpy arrays
    '''
    pose= np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark]).flatten() if result.pose_landmarks else np.zeros(33*4) 
    face= np.array([[res.x, res.y, res.z] for res in result.face_landmarks.landmark]).flatten() if result.face_landmarks else np.zeros(468*3) 
    lh= np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]).flatten() if result.left_hand_landmarks else np.zeros(21*3)
    rh= np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten() if result.right_hand_landmarks else np.zeros(21*3)
    return pose, face, lh, rh
    

def get_frame_coordinates(result, frame):
    '''
    This function turns the result objects to a list of coordinates
    '''
    p_co= [(int(r.x * frame.shape[1]), int(r.y * frame.shape[0])) for r in result.pose_landmarks.landmark] if result.pose_landmarks else [(0,0)]*33 
    f_co= [(int(r.x * frame.shape[1]), int(r.y * frame.shape[0])) for r in result.face_landmarks.landmark] if result.face_landmarks else [(0,0)]* 468 
    l_co= [(int(r.x * frame.shape[1]), int(r.y * frame.shape[0])) for r in result.left_hand_landmarks.landmark] if result.left_hand_landmarks else [(0,0)]*21
    r_co= [(int(r.x * frame.shape[1]), int(r.y * frame.shape[0])) for r in result.right_hand_landmarks.landmark] if result.right_hand_landmarks else [(0,0)]*21
    return p_co, f_co, l_co, r_co

# function to interpolate two frames of a video.
def interpolate_frame_detections(most_recent_detection, next_coming_detection, alpha):
    """
    Based on the value of most recent detection and next coming detection which are the frames before and after our faulty frame returns a landmark array for
    the faulty frame.
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

# function to fill the empty detections in the videos using interpolation
def fill_empty_detections(detections):
    """
    In principle fills up the empty landmark detections for frames that where faulty in the dataset and returns the dataset .
    Args:
        detections: all video detections from mediapipe
    Returns:
        detections (with no empty landmark frame)
    Example use: 
        detections= fill_empty_detections(detections)
    """
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
