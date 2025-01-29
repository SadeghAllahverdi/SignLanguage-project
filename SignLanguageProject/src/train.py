# Explanation: this python file carries out the training process. from making the datast to plotting the results.
#-----------------------------------------------------------------------Import-------------------------------------------------------------------------------

# importing numpy, torch, nn, typing: for writing input types for the functions
import numpy as np 
from numpy.typing import NDArray
from typing import Callable, List, Tuple, Literal
import torch
from torch import nn
# from tqdm.auto import tqdm  

import torch.optim as optim                   #optimizer
from torch.utils.data import Dataset          # dataset calss
from torch.utils.data import DataLoader       #data loader
    
from sklearn.utils import resample         # used for bootstrapping
from sklearn.model_selection import KFold  # for K fold cross validation if necessary

# connecting the steps
from preprocess_utils import interpolate_dataset, split_dataset, convert
from plot_utils import draw_in_tensorboard, plot_confusion_matrix, plot_loss_accuracy
from train_utils import train_model, reset_model_parameters # for K fold cross validation if necessary

#-------------------------------------------------------------Constant variables and classes-------------------------------------------------------------------
# path to experiment directory: where the tensorboard files are saved (!!!!should be changed based on system file structure!!!!)
experiment_dir= "C:/Users/sadeg/OneDrive/Desktop/Thesis/python_codes/SignLanguageProject/experiment_results"
# path to the current directory incase we want to save some plotting pictures quickly (!!!!should be changed based on system file structure!!!!)
current_dir= "C:/Users/sadeg/OneDrive/Desktop/Thesis/python_codes/SignLanguageProject/notebooks"
# A simple dataset class from to CustomImageDataset example from pytorch.org
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
        
#----------------------------------------------------------------------Train functions-------------------------------------------------------------------------
# function to configure the training enviroment
def configure(detections: NDArray[np.float64], 
              labels: List[str],
              class_names: List[str],
              test_size: float,
              batch_size: int,
              num_epochs: int,
              model: torch.nn.Module,
              lr: float,
              device: torch.device,
              quick_save: bool,
              results_name: str,
              dataset_name: Literal['LSA64', 'AUTSL40']):
    """
    This function configures the training enviroment. The function first splits the datasets, then creates dataset and dataloader objects, following that 
    the functions sends the model to the device, it defines loss function and optimizer algorithim for the train process. Afterwards the model trains the model
    and plots the results and saves them to the right directory.
    Args:
        detections: array of all video detections
        labels: list of all video labels
        class_names: a list containing unique class names in the dataset
        batch_size: batch size
        num_epochs: number of epochs
        lr: learning rate
        device: the device that we use for training (Cuda or CPU)
        results_name: used to identify different training results
        quick_save: a boolean for quick saving the plots in current dir
        data_set_dir: The directory where the results are saved
    """
    #X_train, X_test, y_train, y_test = train_test_split(detections, labels, test_size= test_size, random_state= 42, stratify=labels)

    X_train, X_test, y_train, y_test= split_dataset(detections, labels, class_names, test_size)  # split the dataset
    train_dataset= CustomDataset(X_train, y_train)      # train_dataset
    test_dataset= CustomDataset(X_test, y_test)         # test dataset
    train_loader = DataLoader(dataset=train_dataset, batch_size= batch_size, num_workers=0, shuffle=True) # train dataloader 
    test_loader = DataLoader(dataset=test_dataset, batch_size= batch_size, num_workers=0, shuffle=False)  # test dataloader
    model= model.to(device)                            # sending model to device: CUDA or CPU     
    loss_fn = nn.CrossEntropyLoss()                    # cross entropy for loss
    optimizer = optim.Adam(model.parameters(), lr= lr) # Adam optimizer

    train_l, test_l, train_a, test_a, y_trues, y_preds = train_model(num_epochs,model, train_loader, test_loader, optimizer, loss_fn, device)  # train model

    save_path= f"{current_dir}/loss_acc.png" if quick_save else None
    plot_loss_accuracy(train_l, test_l, train_a, test_a, batch_size, save_path)  # loss acc
    save_path= f"{current_dir}/confmat.png" if quick_save else None
    plot_confusion_matrix(y_trues, y_preds, class_names, num_epochs, save_path)  # confusion matrix

    log_dir =f'{experiment_dir}/{dataset_name}/{model.model_type}/runs/{results_name}/'         # directory for saving the tensorboard files
    draw_in_tensorboard(train_l, test_l, train_a, test_a, log_dir)  # drawing in tensor board

# function to configure the Kfold cross validation
def configure_Kfold(detections: NDArray[np.float64], 
                    labels: List[str],
                    class_names: List[str],
                    n_splits: int,
                    batch_size: int,
                    num_epochs: int,
                    model_class: torch.nn.Module,
                    model_args: dict,
                    lr: float,
                    device: torch.device,
                    quick_save: bool):
    """
    This function configures the training enviroment for KFold cross validation. For each fold, The function first splits the datasets, then creates dataset
    and dataloader objects, following that the functions sends the model to the device, it defines loss function and optimizer algorithim for the train 
    process. Afterwards the model trains the model and plots the results and saves them to the right directory. 
    Args:
        detections: array of all video detections
        labels: list of all video labels
        class_names: a list containing unique class names in the dataset
        n_splits: the number of folds that the data will be divided to
        batch_size: batch size
        num_epochs: number of epochs
        lr: learning rate
        device: Cuda or CPU
        quick_save: a boolean for quick saving the plots in current dir
    """
    X, y= convert(detections, labels, class_names) # converting detections and labels to the right format.
    dataset= CustomDataset(X, y)                   # making dataset
    kf= KFold(n_splits=n_splits, shuffle=True)     # making kfold object to split the dataset
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1} ------------------------------------------------------------------------------------------------------------------------------")
        model = model_class(**model_args).to(device)
        train_loader = DataLoader(dataset=dataset, batch_size= batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx))  # train_dataloader
        test_loader = DataLoader(dataset=dataset, batch_size= batch_size, sampler=torch.utils.data.SubsetRandomSampler(test_idx))    # test dataloader
        loss_fn = nn.CrossEntropyLoss()                       # loss function
        optimizer = optim.Adam(model.parameters(), lr= lr)    # optimizer

        train_l, test_l, train_a, test_a, y_trues, y_preds = train_model(num_epochs,model, train_loader, test_loader, optimizer, loss_fn, device) # train model

        save_path= f"{current_dir}/{fold+1}.png" if quick_save else None 
        plot_loss_accuracy(train_l, test_l, train_a, test_a, batch_size, save_path)    # loss acc
        # save_path= f"{current_dir}/c{fold+1}.png" if quick_save else None
        # plot_confusion_matrix(y_trues, y_preds, class_names, num_epochs, save_path)  # confusion matrix
