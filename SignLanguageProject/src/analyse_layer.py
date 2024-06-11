
# This file containes all functions necessary to calculate and analyse layer attributions. The file contians multiple functions for getting
# mediapipe landmarks, drawing , calculation etc. Although the functions are very different in principle and one might argue it is better to
# distribute between other .py files. I have decided to put them all here because they are all used to gether.

import os
import cv2
import torch
from torch import Tensor
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

mp_holistic= mp.solutions.holistic
mp_drawing= mp.solutions.drawing_utils

def get_landmarks(video_path: str,
                  frame_numbers: int = 30):

    """
    This function creates a list of landmarks, list of pixel coordinates(x,y) and a list mediapipe output objects (here called results)
    for a video path by applying mediapipe model frame by frame.
    Args:
        video_path: Path to video.
        frame_numbers: number of frames we want to take from the entire video.
        
    Returns:
        A tuple of (results, pixel_coor, video_detections, label) Where :
        results is a list of mediapipe objects that is later used to visualize landmarks.
        pixel_coor is a list of (px, py) coordinates, this is later used to draw circles on specific landmarks.
        video_detections is an array of flattened mediapipe detections obtained from the video
        label is the video label as an index number (range 0 to 63)

    Note:
        video_detections has the following structure:
        
        indexes (0 to 131) of the list correspond to the first 33 pose landmarks : [x, y, z, visibility]
        indexes (132 to 1535) of the list correspond to the first 468 face landmarks: [x, y, z]
        indexes (1536 to 1598) of the list correspond to the first 21 left hand landmarks: [x, y, z]
        indexes (1599 to 1661) of the list correspond to the first 21 right hand landmarks: [x, y, z]

        in total 1662 coordinate/visibility values OR 543 total landmark objects
    """
    
    #'C:/Users/sadeg/OneDrive/Desktop/Thesis/python_codes/lsa64_raw/all/001_001_001.mp4'
    with mp.solutions.holistic.Holistic(min_detection_confidence= 0.5, min_tracking_confidence=0.5) as holistic:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"ERROR in opening the video path{video_path}")    
        else:
            total_frames_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))                               
            frame_idxs_to_process = np.linspace(0, total_frames_number-1, frame_numbers, dtype=int)  
            results= []
            pixel_coor = []
            video_detections= []
            
            for idx in frame_idxs_to_process:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame= cap.read()
                if not ret:
                    break       
                result= holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # 1. add result obj to results list
                results.append(result)
                
                # convert result obj landmark values to np.array for each frame
                pose= np.array([[res.x, res.y, res.z, res.visibility] for res in result.pose_landmarks.landmark]).flatten() if result.pose_landmarks else np.zeros(33*4) 
                face= np.array([[res.x, res.y, res.z] for res in result.face_landmarks.landmark]).flatten() if result.face_landmarks else np.zeros(468*3) 
                lh= np.array([[res.x, res.y, res.z] for res in result.left_hand_landmarks.landmark]).flatten() if result.left_hand_landmarks else np.zeros(21*3)
                rh= np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten() if result.right_hand_landmarks else np.zeros(21*3)
                detection= np.concatenate((pose,face,lh, rh))
                
                # 2. add landmark values from each frame to video_detection list
                video_detections.append(detection)
                
                # convert result obj landmark values pixel_coordinates list for each frame
                c1= [(int(res.x * frame.shape[1]), int(res.y * frame.shape[0])) for res in result.pose_landmarks.landmark] if result.pose_landmarks else [(0, 0)] * 33 
                c2= [(int(res.x * frame.shape[1]), int(res.y * frame.shape[0])) for res in result.face_landmarks.landmark] if result.face_landmarks else [(0, 0)] * 468 
                c3= [(int(res.x * frame.shape[1]), int(res.y * frame.shape[0])) for res in result.left_hand_landmarks.landmark] if result.left_hand_landmarks else [(0, 0)] * 21
                c4= [(int(res.x * frame.shape[1]), int(res.y * frame.shape[0])) for res in result.right_hand_landmarks.landmark] if result.right_hand_landmarks else [(0, 0)] * 21
                c= c1+c2+c3+c4

                # 3. create list of (px,py) coordinates for video
                pixel_coor.append(c)
                    
            label= int(os.path.basename(video_path).split('_')[0]) - 1
    
        cap.release()
        
        return results, pixel_coor, video_detections, label


def calculate_means(attributions: Tensor):
    """
    Calculates the mean Layer attribution of a landmark. each landmark has x, y, z (and in case of pose_landmarks visibility) values
    each landmark value has an attribution that can effect the transformer layer. mean acts as a parameter that shows how much a 
    landmark is effecting the output of the model.
    Args:
        attributions: a tensor of shape(1, frame_number, 1662)
    Returns:
        means: a tensor of shape (1, frame_number, 543)
    """
    # calculate mean for first 132 pose landmark, pose landmark structure is (x, y, z, visibility)
    first_part = attributions[:, :, :132]                            
    first_part_reshaped = first_part.reshape(attributions.shape[0], attributions.shape[1], -1, 4)
    first_part_means = first_part_reshaped.mean(dim=3)

    # calculate mean for the rest of landmarks (face, left_hand, right_hand), their structure is (x, y, z)
    second_part = attributions[:, :, 132:]
    second_part_reshaped = second_part.reshape(attributions.shape[0], attributions.shape[1], -1, 3)
    second_part_means = second_part_reshaped.mean(dim=3)
    
    # Concatenate first and second part
    means = torch.cat((first_part_means, second_part_means), dim=2)
    
    return means


def make_idx_tr_pairs(indices: Tensor, 
                      means: Tensor):
    """
    Args:
        indices: a tensor that contains indices of most to least significant landmarks for each frame
        means: output of the above function it is basically used as a transparency score from 0 to 10

    Returns:
        idx_tr: an ordered list containing indices of most to least significant landmarks for each frame along with their transparency score

    Note:
        the reason idx_tr is an ordered list is so that we can draw landmarks from least to most important. this is usefull
        when for example a left hand landmark that is important hovers over a face landmrk. in this case it is drawn on top of 
        the less imoprtant landmark.
    """
    #prepare transparency_level
    transparency_level= means/ torch.max(abs(means))
    transparency_level= transparency_level * 10
    
    
    list_1= indices.tolist()
    list_2= transparency_level.int().tolist()
    
    idx_trs = []
    for f in range(len(list_1[0])):  
        frame = []
        for i in range(len(list_1[0][f])):
            index = list_1[0][f][i]  
            value = list_2[0][f][index]  
            frame.append((index, value))  
        idx_trs.append(frame)  

    return idx_trs


def plot_atts_heatmap(attributions: Tensor, title="Attributions"):
    """
    This function plots attribution as a heatmap where x axis are attributes and y axis are frames
    """
    attributions = attributions.detach().cpu().numpy()  # Move tensor to CPU before converting to NumPy
    batch_size, seq_len, num_features = attributions.shape
    
    for i in range(batch_size):
        plt.figure(figsize=(12, 6))
        plt.imshow(attributions[i].T, cmap='viridis', aspect='auto', origin='lower')
        plt.colorbar()
        plt.title(f"{title} - Sample {i}")
        plt.xlabel("Sequence Length")
        plt.ylabel("Features")
        plt.xticks(range(0, seq_len, 1))  # Set ticks every 5 units
        plt.xlim(0, seq_len - 1)  # Set x-axis limits
        plt.yticks(range(0, num_features, 100))  # Set ticks every 10 units
        plt.ylim(0, num_features - 1)  # Set y-axis limits
        plt.show()


def plot_mp_landmarks(frame, result):
    """
    This function draws landmarks and connections on a given frame.
    Args:
        frame: video frame that we want to draw on.
        result: the detected media pipe object corresponding to the frame.
        
    """
    mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color= (0, 255, 0), thickness= 1, circle_radius= 2),
                              mp_drawing.DrawingSpec(color= (0, 255, 0), thickness= 1, circle_radius= 2))
    mp_drawing.draw_landmarks(frame, result.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color= (0, 255, 0), thickness= 1, circle_radius= 2),
                              mp_drawing.DrawingSpec(color= (0, 255, 0), thickness= 1, circle_radius= 2))
    mp_drawing.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color= (0, 255, 0), thickness= 1, circle_radius= 2),
                              mp_drawing.DrawingSpec(color= (0, 255, 0), thickness= 1, circle_radius= 2))
    mp_drawing.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color= (0, 255, 0), thickness= 1, circle_radius= 2),
                              mp_drawing.DrawingSpec(color= (0, 255, 0), thickness= 1, circle_radius= 2))



def plot_circle(frame, coor, idx_tr):
    """
    This function visualizes layer attributions in a frame.
    Args:
        frame: video frame that we want to draw on.
        coor: list of all (x, y) coordinates of the landmarks detected in the frame
        idx_tr: contain indexes of landmarks and their transparency score. the indexes correspond to the 
        coordinates of the landmarks in coor list.
    Note:
        for more information about landmark indexes take a look at get_landmarks and draw_layer_attr functions.
        
    """
    for idx, tr in reversed(idx_tr[:75]):
        intensity = int(min(255, max(0, 255 * abs(tr) / 10)))

        color = (intensity, 255, 0)  

        cv2.circle(frame, (coor[idx][0], coor[idx][1]), radius=5, color=color, thickness=-1)


def draw_layer_attr(video_path, results, pixel_coor, idx_trs, frame_numbers = 30, wait= 200):
    """
    This function visualizes layer attributions in the video.
    Args:
        video_path: path to the video.
        results: set of all (x, y) coordinates of the landmarks detected in the frame
        pixel_coor: this list contains indexes of most important landmarks, it can be any number from 0 to 542.
        idx_trs: an ordered list of tuples containing the index landmarks and their transparency
        
    """
    #'C:/Users/sadeg/OneDrive/Desktop/Thesis/python_codes/lsa64_raw/all/001_001_001.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR in opening the video path{video_path}")
    else:
        total_frames_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idxs_to_process = np.linspace(0, total_frames_number - 1, frame_numbers, dtype=int)
        
        for frame_idx, result, coor, idx_tr in zip(frame_idxs_to_process ,results , pixel_coor, idx_trs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            plot_mp_landmarks(frame, result)
            plot_circle(frame, coor, idx_tr)

            width = int(frame.shape[1] * 0.60)
            height = int(frame.shape[0] * 0.60)
            resized_frame = cv2.resize(frame, (width, height))

            cv2.imshow("Video", resized_frame)
        
            # Set wait time to 33 milliseconds for approx. 30 fps
            if cv2.waitKey(wait) & 0xFF == 27:  # Exit on ESC key
                break
        
        cap.release()
        cv2.destroyAllWindows()
