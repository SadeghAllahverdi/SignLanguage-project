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
- **Evaluation**: Accuracy metrics, saliency maps, and confusion matrices

## 📸 Sample Results  

### **1️⃣ Average Movement Trajectories (AUTSL Subset)**  
<p align="center">
  <img src="images/movements_autsl.png" width="45%">
  <img src="images/movements_lsa64.png" width="45%">
</p> 

### **2️⃣ Performance vs Hyperparameters**  
<p align="center">
  <img src="images/performancevslayers.png" width="30%">
  <img src="images/performancevsnumheads.png" width="30%">
  <img src="images/performancevsdff.png" width="30%">
</p>

### **2️⃣ Average Saliency Maps**  
<p align="center">
  <img src="images/Saliencyautsl.png" width="40%">
  <img src="images/Saliencyautsl.png" width="40%">
</p>

### **3️⃣ Model Architecture**  
![Model Architecture](images/model_architecture.png) 

 

**Please download the datasets from the links below:**
<br/>
<br/>
**LSA64** dataset has 4 versions in 30, 40, 50 and 60 fps along with their labels
<br/>
**Download extracted Mediapipe landmarks and labels**
<br/>
(https://drive.google.com/drive/folders/1AjV780y033Cy9k9PV9Y2RBOndS1sG4Fd?usp=drive_link)
<br/>
**File path:** SignLanguageProject/data/landmarks_lsa64
<br/>
<br/>
<br/>
AUTSL dataset is only provided in 30 fps.
<br/>
**Download extracted Mediapipe landmarks and labels**
<br/>
(https://drive.google.com/drive/folders/1vupDY3DaFvmBdt_beXWIMqShPkHrcVeU?usp=drive_link)
<br/>
**File path:** SignLanguageProject/data/landmarks_autsl40

**After download, please copy the downloaded files under the provided file path.**
