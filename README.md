Facial Emotion Recognition using Deep Learning

🎯 Overview
This project implements multiple machine learning approaches for Facial Emotion Recognition (FER) using the FER-2013 dataset. The solution includes:

CNN-based Deep Learning models for image classification
Traditional ML models (MLP, Random Forest with PCA)
Comprehensive data preprocessing pipeline
Model evaluation and comparison
Visualization tools for data exploration and results
The primary goal is to classify 7 basic human emotions: Anger, Disgust, Fear, Happy, Sad, Surprise, Neutral.

📊 Dataset
Source: FER-2013 (ICML Face Data)
Size: 35,887 grayscale images (48x48 pixels)
Classes: 7 emotions

0: Anger (4,953 samples)
1: Disgust (547 samples) 
2: Fear (5,121 samples)
3: Happy (8,989 samples)
4: Sad (6,077 samples)
5: Surprise (4,002 samples)
6: Neutral (6,198 samples)

Splits: Training (28,709), Public Test (3,589), Private Test (3,589)

✨ Features
✅ Pixel string to image array conversion
✅ Data normalization and preprocessing
✅ Class imbalance visualization
✅ Multiple CNN architectures
✅ PCA dimensionality reduction
✅ Model comparison (CNN vs MLP vs Random Forest)
✅ Confusion matrix visualization
✅ Comprehensive evaluation metrics

🛠️ Models Implemented

Model                 Test Accuracy                Key Features

CNN (Primary)         ~33.7%                 Conv2D layers, MaxPooling, Dropout, Dense


MLP (Flattened)       ~25.4%                 Dense layers on flattened images


CNN (Improved)        ~44.3%                 Optimized architecture


Random Forest + PCA    46.6%                 Best performing** with 50 PCA components

🚀 Installation

Prerequisites
Python 3.8+
Anaconda/Miniconda recommended
1. Clone the repository
   
git clone https://github.com/Nourhanmohamed12/tools_project.git
cd facial-emotion-recognition

2. Create virtual environment
conda create -n fer python=3.9
conda activate fer

4. Install dependencies
   
pip install -r requirements.txt
requirements.txt:

tensorflow==2.15.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
mlxtend>=0.23.0
scikit-plot>=0.3.7

📖 Usage
1. Download Dataset
Place icml_face_data.csv in the data/ directory.

2. Data Exploration

jupyter notebook notebooks/01_data_exploration.ipynb

3. Train CNN Model

python src/train_cnn.py --epochs 20 --batch-size 32

4. Train All Models & Compare

python src/train_all_models.py

5. Evaluate Models

python src/evaluate_models.py
📈 Results
CNN Model Architecture

Layer (type)           Output Shape       Param #
Conv2D (32, 3x3)      (48, 48, 32)       320
Conv2D (32, 3x3)      (48, 48, 32)       9,248
MaxPooling2D (2x2)    (24, 24, 32)       0
Dropout (0.25)        (24, 24, 32)       0
Conv2D (64, 3x3)      (24, 24, 64)       18,496
Conv2D (64, 3x3)      (24, 24, 64)       36,928
MaxPooling2D (2x2)    (12, 12, 64)       0
Dropout (0.25)        (12, 12, 64)       0
Conv2D (128, 3x3)     (12, 12, 128)      73,856
Conv2D (128, 3x3)     (12, 12, 128)      147,584
MaxPooling2D (2x2)    (6, 6, 128)        0
Dropout (0.25)        (6, 6, 128)        0
Flatten               (4,608)            0
Dense (64)            (64)               294,976
Dropout (0.25)        (64)               0
Dense (32)            (32)               2,080
Dropout (0.25)        (32)               0
Dense (7, softmax)    (7)                231
Total params: 583,719
Performance Summary

Model Performance on Test Set:
├── CNN (Primary):           27.19%
├── MLP (Flattened):         25.42%
├── CNN (Improved):          44.27%
└── Random Forest + PCA:     46.63% ⭐


👩‍💻 Author

Nourhan Mohammed
Computer Science Student | Data Enthusiast
