
import sys
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

from sklearn.model_selection import train_test_split                                                
from collections import defaultdict
from typing import Callable, List, Tuple, Literal
from numpy.typing import NDArray

import torch
from torch import nn
import torch.optim as optim                                                          
import torch.nn.functional as F 
import seaborn as sns
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 
from torchinfo import summary
    
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from sklearn.model_selection import KFold
from captum.attr import LayerConductance

from preprocessing import interpolate_dataset, split_dataset, CustomDataset, convert 
from training import train, accuracy_fn
from plotting import draw_in_tensorboard, plot_confusion_matrix, plot_loss_accuracy
from models import reset_model_parameters


def train_parameters(detections: NDArray[np.float64], 
                     labels: List[str],
                     class_names: List[str],
                     test_size: float,
                     random_state: int,
                     batch_size: int,
                     num_epochs: int,
                     model: torch.nn.Module,
                     learning_rate: float,
                     device: torch.device,
                     dir: Literal['LSA64', 'WLASL100']):
    """
    This function sets different parameters for the training of the model.  
    Args:
        detections: represents media pipe landmarks obtained from the dataset
        labels: list of all video labels
        class_names: a list containing unique class names in the dataset
        random_state: used to add consistansy to the results
        batch_size: determines the batch size
        num_epochs: determines how many times we train the model on the dataset
        learning_rate: determines the rate with which we apply changes to  model parameters
        device: Cuda
        
    Returns:
        plots loss and accuracy for both train and test dataset in tensor board. also plots confusion matrix
    """
    
    # Define the path where the results are saved
    save_path =f'C:/Users/sadeg/OneDrive/Desktop/Thesis/python_codes/SignLanguageProject/experiment_results/{dir}/{model.model_type}/runs/'
    
    # Call the function to split the dataset
    xtrain, xtest, ytrain, ytest= split_dataset(detections, labels, class_names, test_size, random_state)

    # Create dataset objects for train and test split
    train_dataset= CustomDataset(xtrain, ytrain)
    test_dataset= CustomDataset(xtest, ytest)

    # Create data loaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size= batch_size, num_workers=0, shuffle=True) 
    test_dataloader = DataLoader(dataset=test_dataset, batch_size= batch_size, num_workers=0, shuffle=False) 

    # Put model on Gpu
    model= model.to(device)                  

    # Define loss and optimizer for the experiment
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr= learning_rate)

    # Call the function to train the model and save the
    train_losses, test_losses, train_accuracies, test_accuracies, y_trues, y_preds = train(num_epochs, model, train_dataloader, test_dataloader, optimizer, loss_fn, accuracy_fn, device)
    
    # Call function to draw the results in tensorboard
    draw_in_tensorboard(train_losses, test_losses, train_accuracies, test_accuracies, save_path)
    
    # Call the function to draw the results
    #plotting.plot_loss_accuracy(train_losses, test_losses, train_accuracies, test_accuracies, batch_size)

    # Call the function to plot the confusion matrix
    plot_confusion_matrix(y_trues, y_preds, class_names, num_epochs)


def train_param_decorator(func):
    def wrapper(detections: NDArray[np.float64], 
                labels: List[str], 
                class_names: List[str], 
                test_size: float, 
                random_state: int, 
                batch_size: int, 
                num_epochs: int,
                model: torch.nn.Module, 
                learning_rate: float, 
                device: torch.device, 
                dir: Literal['LSA64', 'WLASL100'], 
                bootstrap_first=False, 
                interpolate_first=False):
        """
        decorator for train_parameters adds bootstrap_first and interpolate_first arguments.
        bootstrap_first makes sure the dataset is bootstrapped and then interpolated
        interpolate_first makes sure the dataset is interpolated and then bootstrapped

        n_samples and min_interpolations are chosen so that the dataset has length of 4000 regardless
        which operation takes place first.

        note: both bootstrap_first and interpolate_first should not be set to True!
        """
        
        if bootstrap_first:
            detections, labels = resample(detections, labels, n_samples=2700, random_state=random_state)
            detections, labels = interpolate_dataset(detections, labels, alpha= 0.5, min_interpolations= 13)
        elif interpolate_first:
            detections, labels = interpolate_dataset(detections, labels, alpha= 0.5, min_interpolations= 13)
            detections, labels = resample(detections, labels, n_samples=4000, random_state=random_state)
        
        return func(detections, labels, class_names, test_size, random_state, batch_size, num_epochs, model, learning_rate, device, dir)
    
    return wrapper

          
def kfold_cross_validation(detections: NDArray[np.float64], 
                           labels: List[str],
                           class_names: List[str],
                           n_splits: int,
                           batch_size: int,
                           num_epochs: int,
                           model: torch.nn.Module,
                           learning_rate: float,
                           device: torch.device):
    """
    This function performs K_fold_cross_validation.  
    Args:
        detections: represents media pipe landmarks obtained from the dataset
        labels: list of all video labels
        class_names: a list containing unique class names in the dataset
        n_splits: determines the number of folds that the data will be divided to
        batch_size: determines the batch size
        num_epochs: determines how many times we train the model on the dataset
        learning_rate: determines the rate with which we apply changes to  model parameters
        device: Cuda
        
    Returns:
        plots loss and accuracy of both train and test dataset for each fold
    """
    # convert detections and labels to float 32 and long which are suitable for training
    X, y= convert(detections, labels, class_names)
    
    # Create a dataset object
    dataset= CustomDataset(X, y)

    # Define Kfold object to perform kfold cross validation
    kf= KFold(n_splits=n_splits, shuffle=True)

    # For loop that trains the model on each fold
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}")
        print("-------")

        # Rest model parameters before training model on each fold
        reset_model_parameters(model)

        # Define data loader objects
        train_dataloader = DataLoader(dataset=dataset, batch_size= batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx)) 
        test_dataloader = DataLoader(dataset=dataset, batch_size= batch_size, sampler=torch.utils.data.SubsetRandomSampler(test_idx))

        # Define loss and optimizer for training
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr= learning_rate)
        
        # Put model on Gpu
        model= model.to(device)                  

        # Train the model and save results (train loss/accuracy , test loss/accuracy and true/prediction) to variables
        train_losses, test_losses, train_accuracies, test_accuracies, y_trues, y_preds = train(num_epochs, model, train_dataloader, test_dataloader, optimizer, loss_fn, accuracy_fn, device)

        # Call the function to plot the result
        plot_loss_accuracy(train_losses, test_losses, train_accuracies, test_accuracies, batch_size)

        # Call the function to plot the confusion matrix
        #plotting.plot_confusion_matrix(y_trues, y_preds, class_names, num_epochs)


def kfold_decorator(func):
    def wrapper(detections: NDArray[np.float64], 
                labels: List[str],
                class_names: List[str],
                n_splits: int,
                batch_size: int,
                num_epochs: int,
                model: torch.nn.Module,
                learning_rate: float,
                device: torch.device,
                random_state= 42,
                bootstrap_first=False, 
                interpolate_first=False):
        """
        decorator for train_parameters adds bootstrap_first, interpolate_first and random_state variables.
        bootstrap_first makes sure the dataset is bootstrapped and then interpolated
        interpolate_first makes sure the dataset is interpolated and then bootstrapped

        n_samples and min_interpolations are chosen so that the dataset has length of 4000 regardless
        which operation takes place first.

        note: both bootstrap_first and interpolate_first should not be set to True!
        """
        
        if bootstrap_first:
            detections, labels = resample(detections, labels, n_samples=2700, random_state=random_state)
            detections, labels = interpolate_dataset(detections, labels, alpha= 0.5, min_interpolations= 13)
        elif interpolate_first:
            detections, labels = interpolate_dataset(detections, labels, alpha= 0.5, min_interpolations= 13)
            detections, labels = resample(detections, labels, n_samples=4000, random_state=random_state)
        
        return func(detections, labels, class_names, n_splits, batch_size, num_epochs, model, learning_rate, device)
    
    return wrapper  
