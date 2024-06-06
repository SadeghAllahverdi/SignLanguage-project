
import torch
import torch.optim as optim                                                          
import torch.nn.functional as F

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
from torch.utils.tensorboard import SummaryWriter

import os                                                                           
import random 
import json
import numpy as np                                                                  
import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns
from tqdm.auto import tqdm 

from typing import Callable, List
from sklearn.metrics import confusion_matrix
                                                   
import cv2                                                                         
import mediapipe as mp  

from glob import glob                                                               
from pathlib import Path                                                                                                                                                                         
from collections import defaultdict
from typing import Callable, List, Tuple
from numpy.typing import NDArray

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


# function for drawing loss and accuracy of a training session
def plot_loss_accuracy(train_losses: List[float],
                       test_losses: List[float],
                       train_accuracies: List[float],
                       test_accuracies: List[float],
                       batch_size: int):
    """
    Draws loss and accuracy of a training session.
    Args:
        train_losses: list of train losses
        test_losses: list of test losses
        train_accuracies: list of train accuracies
        test_accuracies: list of test accuracies
        batch_size: batch size

    Example usage:
        plot_loss_accuracy(train_losses, test_losses, train_accuracies, test_accuracies, 64)
    """
    plt.figure(figsize=(18, 9))

    # Loss
    plt.subplot(1, 2, 1) 
    plt.plot(train_losses, label='Train Loss')  
    plt.plot(test_losses, label='Test Loss')
    plt.title(f'Loss over Epochs(batch size= {batch_size}), Last Loss:{test_losses[-1]}, ')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title(f'Accuracy over Epochs(batch size= {batch_size}), Last Accuracy: {test_accuracies[-1]}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


# function for drawing confusion matrix
def plot_confusion_matrix(y_trues: List[int],
                          y_preds: List[int],
                          class_names: List[str],
                          num_epochs: int):
    """
    Trains a model on the given train and test data sets.
    Args:
        y_trues: true values
        y_preds: model predictions
        class_names: list of all class names in the dataset
        num_epochs: number of epochs
    Example usage:
        plot_confusion_matrix(y_trues, y_preds, class_names, num_epochs)
    """
    
    conf_matrix = confusion_matrix(y_trues, y_preds)
    plt.figure(figsize=(18, 15))
    
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted classes')
    plt.ylabel('Actual classes')
    
    plt.title(f'Confusion Matrix after {num_epochs} epoches')
    plt.show()


def draw_in_tensorboard(train_losses: List[float],
                        test_losses: List[float],
                        train_accuracies: List[float], 
                        test_accuracies: List[float],  
                        save_directory: str):
    
    with SummaryWriter(log_dir= save_directory) as writer:
        for epoch , (train_l, test_l, train_a, test_a) in enumerate(zip(train_losses, test_losses, train_accuracies, test_accuracies)):
            writer.add_scalar('Loss/train', train_l, epoch)
            writer.add_scalar('Loss/test', test_l, epoch)
            writer.add_scalar('Accuracy/train', train_a, epoch)
            writer.add_scalar('Accuracy/test', test_a, epoch)
