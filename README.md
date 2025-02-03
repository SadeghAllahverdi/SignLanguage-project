# SignLanguage-project
# Explainable Transformer Architecture for Word-Level Sign Language Classification  

## 📌 Project Overview  
This project explores a **transformer-based deep learning model** for **word-level sign language classification**, with a strong emphasis on **model explainability**. The approach utilizes **saliency maps** to analyze and interpret model decisions, improving transparency in predictions.  

## 📊 Key Highlights  
- **📚 Datasets**  
This project uses publicly available sign language datasets:

- **[AUTSL Dataset](https://cvml.ankara.edu.tr/datasets/)**  
  - Sincan, O. M., & Keles, H. Y. (2020). AUTSL: A Large Multi-Variability Dataset for Turkish Sign Language Recognition. *IEEE Transactions on Biometrics, Behavior, and Identity Science*.

- **[LSA-64 Dataset](https://facundoq.github.io/datasets/lsa64/)**  
  - Ronchetti, F., Quiroga, F., Estrebou, C., Lanzarini, L., & Rosete, A. (2016). LSA64: A Dataset for Argentinian Sign Language Recognition. *IEEE Latin America Transactions*.

🔹 If you use this work, please also **cite these datasets** accordingly.
  
- **🛠 Feature Extraction**:  
  - Used **pose estimation** and data preprocessing to extract skeletal motion features.  
  - Reduced videos to **30 frames with 258 features per frame** for efficiency.  
- **📈 Model Performance**:  
  - **AUTSL dataset**: Achieved **86.26% accuracy** using a **2-layer transformer with sinusoidal positional encoding**.  
  - **LSA-64 dataset**: Accuracy ranged between **90-94%** with optimized transformer configurations.  
- **⚡ Comparisons**:  
  - Outperformed **LSTM models** in both **accuracy and training speed**.  
- **🧐 Explainability**:  
  - **Saliency analysis** revealed the model’s focus on **active hand movements** and **mid-frame segments**.  
  - Misclassifications were linked to **overlapping hand trajectories**.  

## 🚀 Technologies Used  
- **Deep Learning Frameworks**: PyTorch  
- **Preprocessing**: MediaPipe for pose detection  
- **Evaluation**: Loss, Accuracy, confusion matrices, Captum's saliency maps, 

## 📸 Sample Results  

### **1️⃣ Average Movement Trajectories**
<div style="display: flex; justify-content: center;">
  <figure style="text-align: center; margin: 10px;">
    <img src="images/movements_autsl.png" width="45%">
    <figcaption>Figure 1: AUTSL Subset</figcaption>
  </figure>
  <figure style="text-align: center; margin: 10px;">
    <img src="images/movements_lsa64.png" width="45%">
    <figcaption>Figure 2: LSA-64 Dataset</figcaption>
  </figure>
</div>

### **2️⃣ Performance vs Hyperparameters**
<div style="display: flex; justify-content: center;">
  <figure style="text-align: center; margin: 10px;">
    <img src="images/performancevslayers.png" width="30%">
    <figcaption>Layers vs Performance</figcaption>
  </figure>
  <figure style="text-align: center; margin: 10px;">
    <img src="images/performancevsnumheads.png" width="30%">
    <figcaption>Attention Heads vs Performance</figcaption>
  </figure>
  <figure style="text-align: center; margin: 10px;">
    <img src="images/performancevsdff.png" width="30%">
    <figcaption>Feedforward Dim vs Performance</figcaption>
  </figure>
</div>

### **3️⃣ Average Saliency Maps**
<div style="display: flex; justify-content: center;">
  <figure style="text-align: center; margin: 10px;">
    <img src="images/saliencyautsl.png" width="40%">
    <figcaption>Figure 6: Saliency Map for AUTSL</figcaption>
  </figure>
  <figure style="text-align: center; margin: 10px;">
    <img src="images/saliencylsa64.png" width="40%">
    <figcaption>Figure 7: Saliency Map for LSA-64</figcaption>
  </figure>
</div>

### **4️⃣ Model Architecture**
<p align="center">
  <img src="images/transformerarchitecture.png" width="60%">
  <figcaption>Figure 8: Transformer-Based Model Architecture</figcaption>
</p>


### **4️⃣ Model Architecture**
<p align="center">
  <figure>
    <img src="images/transformerarchitecture.png" width="60%">
    <figcaption>Figure 8: Transformer-based model architecture for sign language classification.</figcaption>
  </figure>
</p>


 

**You can download the landmarks that were detected using mediapipe here:**
<br/>
<br/>
**LSA64** landmarks were extracted in 4 versions in 30, 40, 50 and 60 fps
<br/>
**Download extracted Mediapipe landmarks and labels**
<br/>
(https://drive.google.com/drive/folders/1AjV780y033Cy9k9PV9Y2RBOndS1sG4Fd?usp=drive_link)
<br/>
**File path:** SignLanguageProject/data/landmarks_lsa64
<br/>
<br/>
<br/>
AUTSL landmarks are only provided in 30 fps.
<br/>
**Download extracted Mediapipe landmarks and labels**
<br/>
(https://drive.google.com/drive/folders/1vupDY3DaFvmBdt_beXWIMqShPkHrcVeU?usp=drive_link)
<br/>
**File path:** SignLanguageProject/data/landmarks_autsl40

**After download, please copy the downloaded files under the provided file path.**
