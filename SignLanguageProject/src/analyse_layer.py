# Explanation: this python file contains functions for analysing layer attentions and drawing sailency maps.
#------------------------------------------------------------------------Import--------------------------------------------------------------------------------
import captum
from captum.attr import Attribution
from captum.attr import Saliency
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance

import os
import cv2
import torch
from tqdm.auto import tqdm 
from torch import Tensor
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from prepare_datasets import get_frame_detections, get_frame_coordinates, autslclass_names, lsa64class_names
from preprocess_utils import convert

#------------------------------------------------------------------------Constants-----------------------------------------------------------------------------
# these parameters most only be used when the 468 extra face landmarks are not included !!!!
pose_idx, lh_idx, rh_idx= list(range(0,33)), list(range(33, 54)), list(range(54, 75))    # indexes in pose, lh, rh
pose_l_idx, pose_r_idx= [11, 13, 15, 17, 19, 21], [12, 14, 16, 18, 20, 22]               # indexes corresponding to right arm body and left arm 
pose_m_idx= [idx for idx in pose_idx if idx not in pose_r_idx and idx not in pose_l_idx] # rest of the indexes: legs, hips, etc
reordered_idxs= pose_m_idx + pose_l_idx + lh_idx + pose_r_idx + rh_idx                   # so left arm lh and right arm rh are next to each other

reordered_landmarks= ["Nose", "Left Eye Inner", "Left Eye", "Left Eye Outer", "Right Eye Inner", "Right Eye", "Right Eye Outer", "Left Ear", "Right Ear",
                      "Left Mouth Corner", "Right Mouth Corner", "Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle", "Left Heel",
                      "Right Heel", "Left Foot Index", "Right Foot Index", "Left Shoulder", "Left Elbow", "Left Wrist", "Left Pinky", "Left Index", 
                      "Left Thumb", "Left Wrist", "Left Thumb CMC", "Left Thumb MCP", "Left Thumb IP", "Left Thumb Tip", "Left Index MCP", "Left Index PIP",
                      "Left Index DIP", "Left Index Tip", "Left Middle MCP",  "Left Middle PIP", "Left Middle DIP", "Left Middle Tip", "Left Ring MCP",
                      "Left Ring PIP", "Left Ring DIP", "Left Ring Tip", "Left Pinky MCP", "Left Pinky PIP", "Left Pinky DIP", "Left Pinky Tip", 
                      "Right Shoulder", "Right Elbow", "Right Wrist", "Right Pinky", "Right Index", "Right Thumb", "Right Wrist", "Right Thumb CMC", 
                      "Right Thumb MCP", "Right Thumb IP", "Right Thumb Tip", "Right Index MCP", "Right Index PIP", "Right Index DIP", "Right Index Tip",
                      "Right Middle MCP", "Right Middle PIP", "Right Middle DIP", "Right Middle Tip",  "Right Ring MCP", "Right Ring PIP", "Right Ring DIP",
                      "Right Ring Tip", "Right Pinky MCP", "Right Pinky PIP", "Right Pinky DIP", "Right Pinky Tip"]

mp_holistic= mp.solutions.holistic         # mediapipe holistic model
mp_drawing= mp.solutions.drawing_utils     # pre-made class that has functions for drawing media pipe result object

#--------------------------------------------------------------plotting data on video sample-------------------------------------------------------------------
def plot_attributions_on_video(video_path: str,
                               model: torch.nn.Module,
                               captum_method: Attribution,
                               class_names: List[str],
                               device: torch.device,
                               frame_numbers: int = 30):
    """
    This function plots the attribution values for a given video (from LSA64 and AUTSL) 
    Args:
        video_path: Path to the video sample.
        captum_method: ex:  Saliency
        class_names: List of all words in the dataset from which we took that video.
        frame_numbers: number of frames we want to take from the entire video.
    Example usage:
        result_objs, coordiantes, video_detection, label= get_landmarks_from_vid(video_path, class_name, 30) 
    """
    vid_idx_to_label= {i:label for i, label in enumerate(class_names)}          # this mapping is used to change the video titles to labels
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR in opening the video path{video_path}")
        return
        
    with mp.solutions.holistic.Holistic(min_detection_confidence= 0.5, min_tracking_confidence=0.5) as holistic:
        try:
            frames= []
            result_objs= []        # stores mediapipe result objects from each frame of the video
            video_detection= []    # stores the detections from each frame of the video
            video_coordinates= []  # stores the coordinates of the detected landmarks
            
            total_frames_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)                                 
            total_frames_number= int(total_frames_number)
            frame_idxs_to_process = np.linspace(0, total_frames_number-1, frame_numbers, dtype=int) # desired frame indexes
            
            for idx in frame_idxs_to_process:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)       # set cv2 to the desired index
                ret, frame= cap.read()                      # process the frame in that index
                if not ret:
                    print("unreadble frame detected")       # incase there is any unreadable frame
                    break   

                result= holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # processing frame with Mediapipe
                pose,face,lh, rh= get_frame_detections(result)
                frame_detection= np.concatenate((pose, lh, rh))                   # here we eliminate face landmarks for drawing.
                p_co, f_co, l_co, r_co= get_frame_coordinates(result, frame)      # turning results into coordinates
                frame_coordinates= p_co+ l_co+ r_co                               # shape: 1, frame_number, 75 tuples                 
                
                frames.append(frame)                                              # append
                result_objs.append(result)                                       
                video_detection.append(frame_detection)  
                video_coordinates.append(frame_coordinates)
                    
            if class_names== autslclass_names:    # for AUTSL
                video_idx= int(os.path.basename(os.path.dirname(video_path))) # extract video index from the video folder
                label= vid_idx_to_label[video_idx]                            # map the video index to a label      
            elif class_names== lsa64class_names:  # for LSA64
                video_idx= int(os.path.basename(video_path).split('_')[0])    # extract index from the video title: 001_004_003 -> 1
                label= vid_idx_to_label[video_idx-1]                          # map the index to the correct label

            print(f"Calculating landmark attributions for the sign language video {label}")
            video_detection = np.array(video_detection, dtype=np.float64)                     # torch said to change list of numpy to np array so its faster
            video_detection, label= convert(video_detection, [label], class_names)                                     # converting to tensors
            attributions= landmark_attributions(model, captum_method, video_detection, label, device)                  # shape: 1, frame_number, 75

            vid_color_scale= (attributions[0] - attributions[0].min()) / (attributions[0].max()-attributions[0].min()) # normalizing attributions
            vid_color_scale= vid_color_scale* 255 
            vid_color_scale = vid_color_scale.cpu().numpy()   

            for i, frame in enumerate(frames):
                plot_mp_landmarks(frame, result_objs[i])
                plot_circle(frame, video_coordinates[i], vid_color_scale[i])
                
                cv2.imshow('Frame with Attributions', frame)
                #cv2.resizeWindow('Frame with Attributions', int(frame.shape[1] * 0.6), int(frame.shape[0] * 0.6))  # make it smaller
                if cv2.waitKey(30) & 0xFF == 27:
                    break

        except Exception as e:
            print(f"Error happened while processing: {e}")
            raise
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
#------------------------------------------------------------------working with attention values-------------------------------------------------------------
# function for calculating mean attentions
def landmark_attributions(model: torch.nn.Module,
                          captum_method: Attribution,
                          video_detection: torch.Tensor, 
                          label: torch.Tensor, 
                          device: torch.device,):
    """
    Calculates the mean layer attribution of landmarks in  a video. each landmark has x, y, z (and in case of pose_landmarks visibility) values. mean layer
    attribution
    acts as a parameter that shows how much a landmark is effecting the output of the model.
    Args:
        model: model that we want to analyse
        captum_method: LayerConductance, Saliency, IntegratedGradients
        video detection : a tensor of shape(frame_number, 1662 or 258)
        label: label of the video
    Returns:
        means: a tensor of shape (1, frame_number, landmark_number)
    """
    model.eval()                                                             # set model to evaluation mode
    model.to(device)
    video_detection, label = video_detection.to(device), label.to(device)    # put on GPU
    video_detection= video_detection.unsqueeze(0)                        
    video_detection.requires_grad_()

    attributions= captum_method.attribute(inputs= video_detection, target= label.item())
    
    pose = attributions[:, :, :132]                                          # shape: 1, frame_number, 132
    pose = pose.reshape(attributions.shape[0], attributions.shape[1], -1, 4) # shape: 1, frame_number, 33, 4
    pose_means = pose.mean(dim=3)                                            # shape: 1, frame_number, 33
    
    rest = attributions[:, :, 132:]                                          # shape: 1, frame_number, 1530 or 126
    rest = rest.reshape(attributions.shape[0], attributions.shape[1], -1, 3) # shape: 1, frame_number, 510 or 42, 3
    rest_means = rest.mean(dim=3)                                            # shape: 1, frame_number, 510 or 42
    means = torch.cat((pose_means, rest_means), dim=2)                       # shape: 1, frame_number, 510+33 or 42+33
    
    return means

def landmark_attributions_for_dataset(model, captum_method, dataset, device):
    '''
    This function calculates the mean layer attribution of landmarks over the dataset.
    '''
    total_lm_atts= None
    for data in tqdm(dataset):
        video_detection, label = data[0], data[1]
        video_lm_atts= landmark_attributions(model, captum_method, video_detection, label, device)
    
        if total_lm_atts is None:
            total_lm_atts = torch.zeros_like(video_lm_atts)
    
        total_lm_atts+= video_lm_atts
    
    lm_atts_dataset = total_lm_atts / len(dataset)
    return lm_atts_dataset
#-------------------------------------------------------------------plotting heatmap -------------------------------------------------------------------------

# function for drawing the attention heatmap
def plot_atts_heatmap(attributions: Tensor, save_path: str, show_landmark_names: bool = False):
    """
    For a video, this function plots attribution as a heatmap where x axis are frames, y axis are landmarks and the colors represent attribution values.
    Args:
        attributions: a tensor of shape(1, frame_number, 1662 or 258)
        show_landmark_names: boolean variable for showing landmark names
    Note: vmin and vmax are chosen based on trail and error.
    """
    atts = attributions.detach().cpu().numpy()     # moving tensor to CPU for drawing 
    num_frames, num_features = atts[0].shape       # remove batch dimention and get frame and featur numbers
    plt.figure(figsize=(20, 15))
    if show_landmark_names:
        atts = atts[:, :, reordered_idxs]          # reorder the landmarks
        y_ticks_labels = reordered_landmarks
        y_ticks_positions = range(0, num_features)
    else:
        y_ticks_labels = None 
        y_ticks_positions = range(0, num_features, 5) 
        
    plt.imshow(atts[0].T, cmap='viridis', aspect='auto', origin='lower') # we transpose attributions so landmarks are on y axis
    plt.colorbar()
    plt.xlim(0, num_frames - 1)
    plt.xticks(range(0, num_frames))
    plt.xlabel("frames")
    plt.ylim(0, num_features - 1)
    plt.yticks(y_ticks_positions, y_ticks_labels)
    plt.ylabel("features")
    plt.title("Attributions")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.tight_layout()
    plt.show()

#---------------------------------------------------------------------plotting on data on video---------------------------------------------------------------
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


def plot_circle(frame, coordinates, frame_colorscale):
    """
    This function visualizes layer attributions by drawing circles on detected landmarks.
    Args:
        frame: The video frame we want to draw on.
        coordinates: List of (x, y) coordinates of landmarks detected in the frame.
        attributions: List of tuples where each tuple is (landmark index, attribution score). The attribution score determines the intensity of the circle
        color.
                      
    """
    for idx, color_scale in enumerate(frame_colorscale):  
        intensity = int(color_scale)
        color = (intensity, 255 , 0)
        x, y = coordinates[idx]
        cv2.circle(frame, (x, y), radius=5, color=color, thickness=-1)
