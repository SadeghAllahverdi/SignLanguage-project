#Explanation: This python file contains functions for implementing the training step
#-------------------------------------------------------------------------Import-------------------------------------------------------------------------------

#importing libraries for training models
import torch  
from torch.utils.data import DataLoader # for writing input types
# importing tqdm for progression bar
from tqdm.auto import tqdm 
# importing typing for writing input types for the functions
from typing import Callable, List

#----------------------------------------------------------------Functions for training a model----------------------------------------------------------------
#function for resetting the model parameters if needed
def reset_model_parameters(model):
    for name, module in model.named_children():
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
            
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
    y_preds= torch.argmax(y_logits, 1)                 # gives the position --> label of the strongest prediction
    corrects= (y_preds==y)                             # compare prediction with truth
    accuracy= corrects.sum().item()/ corrects.shape[0] # number of true predictions / all predictions
    return accuracy

# function to train the model
def train_model(num_epochs: int,
                model: torch.nn.Module,
                train_dataloader: DataLoader,
                test_dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.Module,
                device: torch.device):
    """
    Trains a model on given train and test data. and returns avg loss and avg accruacies for each epoch.
    Args:
        num_epochs: number of times (epochs) the model is trained with the entire dataset
        model: model object
        train_dataloader: DataLoader object of train dataset
        test_dataloader: DataLoader object of test dataset.
        optimizer: optimizing entity that updates the weights of the model
        loss_fn: function to calculate loss
        device: Cuda or CPU
    Returns:
        A tuple of (train_losses, test_losses, train_accuracies, test_accuracies, y_trues, y_preds) where:
        train_losses is a list that contains avg train loss of all batches, for every epoch.
        test_losses is a list that contains avg test loss of all batches, for every epoch.
        train_accuracies is a list that contains avg train accuracy of all batches, for every epoch.
        test_accuracies is a list that contains avg test accuracy of all batches, for every epoch.
        y_trues and y_preds are used to draw confusion matrix (they get overwritten in each epoch so in principle the last value of y_trues and y_preds is
        returned).
    Example usage: 
        results= train(num_epochs, model, train_dataloader, test_dataloader, optimizer, loss_fn, accuracy_fn, device)
        train_losses, test_losses, train_accuracies, test_accuracies, y_trues, y_preds= results[0], results[1], results[2], results[3], results[4], results[5]
    """
    
    train_losses= []     
    test_losses= []          
    train_accuracies= []      
    test_accuracies= []       
 
    for epoch in tqdm(range(num_epochs), desc="Training Epoch"):
        model.train()
        train_loss= [] # a list to store loss of every batch
        train_acc= []  # a list to store acc of every batch

        for X, y in train_dataloader:
            # sending detections and labels to device
            X= X.to(device) 
            y= y.to(device)

            # train the model
            optimizer.zero_grad()
            y_logits = model(X)
            loss = loss_fn(y_logits, y)        # batch loss
            loss.backward()
            optimizer.step()

            accuracy= accuracy_fn(y_logits, y) # batch accuracy

            #add loss and accuray of the batch to the list
            train_loss.append(loss.item())
            train_acc.append(accuracy)
            
        # adding average loss and accuracy for the epoch
        train_losses.append(sum(train_loss) / len(train_loss))  
        train_accuracies.append(sum(train_acc) / len(train_acc))
    
        model.eval()      # setting model to evaluation mode so no weights are changed

        y_trues= []       
        y_preds= []       
        test_loss= []     # list to store loss of every batch
        test_acc= []      # list to store accuracy of every batch
        
        with torch.no_grad():
            for X, y in test_dataloader:
                X = X.to(device)
                y = y.to(device)
                
                y_logits = model(X)
                loss = loss_fn(y_logits, y)        # test batch loss
                accuracy= accuracy_fn(y_logits, y) # test batch accuracy
                
                test_loss.append(loss.item())
                test_acc.append(accuracy)
                y_pred= torch.argmax(y_logits, 1)                 # predicted labels
                
                y_trues.extend(y.flatten().cpu().numpy())          # Store true labels
                y_preds.extend(y_pred.flatten().cpu().numpy())     # Store predictions
                
        test_losses.append(sum(test_loss) / len(test_loss))
        test_accuracies.append(sum(test_acc) / len(test_acc))

    return train_losses, test_losses, train_accuracies, test_accuracies, y_trues, y_preds
