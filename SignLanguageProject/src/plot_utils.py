# Explanation: This python file contains functions for plotting training results and other important data.
#----------------------------------------------------------------------Import-----------------------------------------------------------------------------------

# importing libraries for plotting data                                                 
import cv2 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix

# for writing input types for the functions                                                                                
from typing import List
from numpy.typing import NDArray

#-----------------------------------------------------------------visualizing video detections-----------------------------------------------------------------
# functions for drawing video landmarks
def draw_circles(frame: np.ndarray,
                 frame_detection: NDArray[np.float64],
                 frame_structure: List[tuple]):
    """
    This function draws circles on the frame based on x and y position of the landmark. it uses the frame_structure list to handle drawing pose landmarks since
    unlike other landmarks they have x, y, z and "visibility values"
    Args:
        frame: represents frame that is shown
        frame_detection: represents the coordinates of the landmarks in a frame that was processed by mediapipe.
        frame_structure: a list that represents the start and end index for each landmark class: pose, face, lh, rh.
    Returns:
        manipulated frame 
    Example usage:
        frame = draw_circles(frame, frame_detection, [(0, 132), (132, 1536), (1536, 1599), (1599, 1662)])
    """
    for (start, end) in frame_structure:                            # iterate through frame structure: pose, face, lh, rh
        bodypart= frame_detection[start:end]               
        if (start, end) == frame_structure[0]:                      # for the pose landmarks:
            for i in range(0, len(bodypart), 4):                    # iterate through pose landmark
                x, y = bodypart[i], bodypart[i+ 1]                  # getting x and y values for drawing the circles
                px = int(x * frame.shape[1])                                                   
                py = int(y * frame.shape[0])
                cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)     # plotting circles on the frame
        else:
            for i in range(0, len(bodypart), 3):
                x, y = bodypart[i], bodypart[i+ 1]
                px = int(x * frame.shape[1]) 
                py = int(y * frame.shape[0])
                cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)
    return frame

# function to show the vidoe detections
def show_video_detections(video_detection: NDArray[np.float64]):
    """
    This function draws Mediapipe landmarks that were detected from a video. It uses a video_detection array that has x, y, z( and visibility for pose) values. 
    here we only focus on the (x,y) coordinates we do not draw in 3D (no z or visibility).
    Args:
        video_detection: an array that represents video detections 
    """
    height, width= 720, 1280
    frame_structure= [(0, 132), (132, 1536), (1536, 1599), (1599, 1662)]    # used to seperate pose, face, lh, rh values when drawing them
    cv2.namedWindow("video detection", cv2.WINDOW_NORMAL)                   # make a window
    cv2.resizeWindow("video detection", width= width, height= height)       # resize the window to desired hight and width
    try:      # try to plot
        for frame_detection in video_detection:
            frame = np.zeros((height, width, 3), dtype=np.uint8)            # making empty black frame
            frame = draw_circles(frame, frame_detection, frame_structure)   # drawing circles on the frame
            cv2.imshow("video detection", frame)
            if cv2.waitKey(100) & 0xFF == 27:  #ESC key
                break
    finally:  # guarantees that destroyAllWindows() is executed at the end. even if there is error in try part
        cv2.destroyAllWindows()

#---------------------------------------------------------------Visualizing training results--------------------------------------------------------------------
def plot_loss_accuracy(train_losses: List[float],
                       test_losses: List[float],
                       train_accuracies: List[float],
                       test_accuracies: List[float],
                       batch_size: int,
                       save_path: str):
    """
    Draws loss and accuracy of a training session.
    Example usage:
        plot_loss_accuracy(train_losses, test_losses, train_accuracies, test_accuracies, 64)
    """
    plt.figure(figsize=(18, 9))
    # plotting loss
    plt.subplot(1, 2, 1) 
    plt.plot(train_losses, label='Train Loss')      
    plt.plot(test_losses, label='Test Loss')
    plt.title(f'Loss over Epochs(batch size= {batch_size}), Last Loss:{test_losses[-1]}') # writing the final loss value
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # plotting accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title(f'Acc over Epochs(batch size= {batch_size}), Last Acc: {test_accuracies[-1]}') # writing the final accuracy
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_trues: List[int],
                          y_preds: List[int],
                          class_names: List[str],
                          num_epochs: int,
                          save_path: str):
    """
    Plots confusion matrix of a model using true values and model predictions.
    Example usage:
        plot_confusion_matrix(y_trues, y_preds, class_names, num_epochs)
    """
    conf_matrix = confusion_matrix(y_trues, y_preds)
    plt.figure(figsize=(18, 15))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix after {num_epochs} epoches')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

# function for drawing in tensor board.
def draw_in_tensorboard(train_losses: List[float],
                        test_losses: List[float],
                        train_accuracies: List[float], 
                        test_accuracies: List[float],  
                        log_dir: str):
    """
    Plots loss and accuracy of the training process in tensor board.
    Example usage:
        draw_in_tensorboard(train_losses, test_losses, train_accuracies, test_accuracies, save_directory)
    """
    with SummaryWriter(log_dir= log_dir) as writer:
        losses_and_accuracies= zip(train_losses, test_losses, train_accuracies, test_accuracies)
        for epoch , (tr_losses, te_losses, tr_accs, te_accs) in enumerate(losses_and_accuracies):
            writer.add_scalar('Loss/train', tr_losses, epoch)
            writer.add_scalar('Loss/test', te_losses, epoch)
            writer.add_scalar('Accuracy/train', tr_accs, epoch)
            writer.add_scalar('Accuracy/test', te_accs, epoch)
            
