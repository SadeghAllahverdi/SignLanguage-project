#importing libraries for training models
import torch
import torch.optim as optim                                                          
import torch.nn.functional as F
import seaborn as sns
import numpy as np                                                                  
import pandas as pd
import matplotlib.pyplot as plt 
from tqdm.auto import tqdm 

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader     
from typing import Callable, List
from sklearn.metrics import confusion_matrix

# function to calculate accuracy
def accuracy_fn(y_logits: torch.Tensor, y: torch.Tensor):
    """
    returns accuracy based on true and predicted label values
    Args:
        y_logits: torch tensor that represents model outputs
        y: torch tensor that represents true output values
    Returns:
        accuracy

    Example usage: 
        accuracy= accuracy_fn(y_logits, y)
    """
    y_preds= torch.argmax(y_logits, 1)
    corrects= (y_preds==y)
    accuracy= corrects.sum().item()/ corrects.size(0)
    return accuracy


# function to train the model
def train(num_epochs: int,
          model: torch.nn.Module,
          train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          accuracy_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
          device: torch.device):
    """
    Trains a model on the given train and test data sets.
    Args:
        num_epochs: number of epoches the model is trained
        model: model object
        train_dataloader: DataLoader object that enables easy access to the samples of train dataset
        test_dataloader: DataLoader object that enables easy access to the samples of test dataset.
        optimizer: represents optimizing algorithms.
        loss_fn: function that calculates loss
        accuracy_fn: function that calculates accuracy
        device: either Cuda or Cpu based on settings
        
    Returns:
        A tuple of (train_losses, test_losses, train_accuracies, test_accuracies, y_trues, y_preds).
        Where :
        train_losses is a list losses over based on train dataset over all epochs.
        test_losses is a list losses over based on test dataset over all epochs.
        train_accuracies is a list accuracies over based on train dataset over all epochs.
        test_accuracies is a list accuracies over based on test dataset over all epochs.
        y_trues, y_preds are used to draw confusion matrix (they are overwritten in each epoch so in principle the value of
        of y_trues and y_preds is returned for the last epoch).

    Example usage: 
        results= train(num_epochs, model, train_dataloader, test_dataloader, optimizer, loss_fn, accuracy_fn, device)
    """
    
    train_losses= []          # stores avg train losses of epoch
    test_losses= []           # stores avg test losses of epoch 
    train_accuracies= []      # stores avg train acc of epoch
    test_accuracies= []       # # stores avg test acc of epoch 
 
    for epoch in tqdm(range(num_epochs), desc="Training Epoch"):
        model.train()
        train_loss= [] # stores avg loss of batch
        train_acc= []  # stores avg acc of batch

        for X, y in train_dataloader:
            X= X.to(device) 
            y= y.to(device)
            
            optimizer.zero_grad()
            y_logits = model(X)
            loss = loss_fn(y_logits, y)        # loss of the batch
            loss.backward()
            optimizer.step()

            accuracy= accuracy_fn(y_logits, y) # accuracy of the batch
            
            train_loss.append(loss.item())
            train_acc.append(accuracy)
        
        train_losses.append(sum(train_loss) / len(train_loss))
        train_accuracies.append(sum(train_acc) / len(train_acc))
    
        model.eval()

        y_trues= []       # stores all the true labels for conf matrix
        y_preds= []       # stores all the predictions for conf matrix
        test_loss= []       # avg batch loss in test data per epoch
        test_acc= []        # avg batch acc in test data per epoch
        
        with torch.no_grad():
            for X, y in test_dataloader:
                X = X.to(device)
                y = y.to(device)
                
                y_logits = model(X)
                loss = loss_fn(y_logits, y)
                accuracy= accuracy_fn(y_logits, y)
                
                test_loss.append(loss.item())
                test_acc.append(accuracy)
                y_pred= torch.argmax(y_logits, 1)                  # predicted labels
                
                y_trues.extend(y.view(-1).cpu().numpy())           # Store true labels
                y_preds.extend(y_pred.view(-1).cpu().numpy())     # Store predictions
                
        
        test_losses.append(sum(test_loss) / len(test_loss))
        test_accuracies.append(sum(test_acc) / len(test_acc))

    return train_losses, test_losses, train_accuracies, test_accuracies, y_trues, y_preds


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
