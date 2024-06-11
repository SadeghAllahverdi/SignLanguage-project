
import os
import cv2
import mediapipe as mp
import numpy as np

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



def draw_landmarks(frame, result):
    """
    This function draws landmarks and connections on a given frame.
    Args:
        frame: video frame that we want to draw on.
        result: the detected media pipe object corresponding to the frame.
        
    """
    mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color= (0, 0, 255), thickness= 1, circle_radius= 1),
                              mp_drawing.DrawingSpec(color= (254, 254, 0), thickness= 1, circle_radius= 1))
    mp_drawing.draw_landmarks(frame, result.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color= (0, 0, 255), thickness= 1, circle_radius= 1),
                              mp_drawing.DrawingSpec(color= (254, 254, 0), thickness= 1, circle_radius= 1))
    mp_drawing.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color= (0, 0, 255), thickness= 1, circle_radius= 1),
                              mp_drawing.DrawingSpec(color= (254, 254, 0), thickness= 1, circle_radius= 1))
    mp_drawing.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color= (0, 0, 255), thickness= 1, circle_radius= 1),
                              mp_drawing.DrawingSpec(color= (254, 254, 0), thickness= 1, circle_radius= 1))



def draw_circle(frame, coor, landmark_att):
    """
    This function visualizes layer attributions in a frame.
    Args:
        frame: video frame that we want to draw on.
        coor: set of all (x, y) coordinates of the landmarks detected in the frame
        layer_att: this list contains indexes of most important landmarks, it can be any number from 0 to 542.
        ! for more information about landmark indexes take a look at get_landmarks and draw_layer_attr functions.
        
    """
    colors = [(0, 100, 0),         # Dark Green
              (34, 139, 34),       # Forest Green
              (50, 205, 50),       # Lime Green
              (0, 250, 154),       # Medium Spring Green
              (144, 238, 144)]     # Light Green
    
    # here we sure that the most important landmark is drawn last.
    # this will help if an important landmark and a less important landmark overlap
    # ex: if the Signer puts his/her hand on theri face.
    for i, idx in enumerate(reversed(landmark_att)):

        # i is used to determine the shade the more important the brighter
        shade_index = min(i // 20, len(colors) - 1)
        color = colors[shade_index]
        
        cv2.circle(frame, (coor[idx][0], coor[idx][1]), 5, color, -1)



def draw_layer_attr(video_path, results, pixel_coor, landmark_atts, frame_numbers = 30, wait= 200):
    """
    This function visualizes layer attributions in the video.
    Args:
        video_path: path to the video.
        results: set of all (x, y) coordinates of the landmarks detected in the frame
        pixel_coor: this list contains indexes of most important landmarks, it can be any number from 0 to 542.
        landmark_atts: a list of landmark indexes where index can be any number from 0 to 542 
        the position of the landmark index in the list determines its importance 
            ex:
            in this frame the landmark 17 has the most attribut
            [17, 22, 11, 19, 444, 242, 22, 111, ... , 44]
            .
            .
            .
            [511, 515, 500, 19, 444, 242, 22, 111, ... , 222]
            in this frame above, landmark 222 has the least attribut
        frame_numbers: number of frames in the video that were used as input for media pipe
        ! for more information about landmark indexes take a look at get_landmarks function.
        
    """
    
    #'C:/Users/sadeg/OneDrive/Desktop/Thesis/python_codes/lsa64_raw/all/001_001_001.mp4'
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR in opening the video path{video_path}")
    else:
        total_frames_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idxs_to_process = np.linspace(0, total_frames_number - 1, frame_numbers, dtype=int)
        
        for frame_idx, result, coor, landmark_att in zip(frame_idxs_to_process ,results , pixel_coor, landmark_atts):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            draw_landmarks(frame, result)
            draw_circle(frame, coor, landmark_att)
            # Display the frame
            cv2.imshow("Video", frame)
        
            # Set wait time to 33 milliseconds for approx. 30 fps
            if cv2.waitKey(wait) & 0xFF == 27:  # Exit on ESC key
                break
        
        cap.release()
        cv2.destroyAllWindows()
