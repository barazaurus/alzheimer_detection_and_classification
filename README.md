# Alzheimer's MRI Classification System

## Overview

This project develops an advanced deep learning system for detecting and classifying Alzheimer's disease stages using brain MRI images. The application leverages multiple neural network architectures to provide accurate and rapid screening of MRI scans, potentially assisting medical professionals in early diagnosis.

## Key Features

- Classify brain MRI images into four distinct Alzheimer's stages:
  - Non Demented
  - Very Mild Demented
  - Mild Demented
  - Moderate Demented

- Multiple Model Architectures:
  - Custom Convolutional Neural Network (CNN)
  - VGG16
  - InceptionV3
  - ResNet50

- High Accuracy Classification
  - Custom CNN: 99.53% accuracy
  - VGG16: 92.02% accuracy
  - ResNet50: 84.04% accuracy
  - InceptionV3: 75.27% accuracy

## System Architecture

### Deep Learning Models
The system implements four distinct neural network architectures:

1. **Custom CNN**
   - Specialized architecture designed specifically for MRI image classification
   - Consists of multiple convolutional layers with increasing filter depths
   - Includes dropout layers for regularization
   - Achieved highest accuracy at 99.53%

2. **Transfer Learning Models**
   - VGG16
   - InceptionV3
   - ResNet50
   - Pre-trained on ImageNet, fine-tuned for Alzheimer's classification

### Technical Stack
- Python
- TensorFlow
- Keras
- Tkinter (GUI)
- NumPy
- Pandas
- Matplotlib

## Dataset

- Total Images: 6,400 brain MRI scans
- Class Distribution:
  - Non Demented: 3,200 images (50%)
  - Very Mild Demented: 2,240 images (35%)
  - Mild Demented: 896 images (14%)
  - Moderate Demented: 64 images (1%)

## Installation

### Prerequisites
- Python 3.8+
- TensorFlow
- Keras
- NumPy
- Pandas
- Tkinter

### Clone the Repository
```bash
git clone https://github.com/yourusername/alzheimer-classification.git
cd alzheimer-classification
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application
```bash
python ClassificationAPP.py
```

### How to Use
1. Launch the application
2. Click "Choose MRI" button
3. Select an MRI image
4. View classification results from multiple models

## Model Performance

### Confusion Matrix Insights
- Best performance in classifying Non-Demented cases
- Custom CNN shows superior ability to distinguish between Very Mild and Non-Demented stages

### Key Metrics
- Precision: Ranges from 0.84 to 1.00
- Recall: Ranges from 0.78 to 1.00
- F1-Score: Ranges from 0.81 to 0.99

## Future Work
- Try more datasets or augment existing data
- Implement visualization of decision-making regions in MRI
- Develop web-based platform
- Extensive clinical validation

## Ethical Considerations
This system is designed to assist medical professionals and should not replace comprehensive medical diagnosis.

## Authors
Andrei-Alexandru Baractaru
Supervised by: Conf. Dr. Anca Ignat

## Acknowledgments
- Alexandru Ioan Cuza University of Iași
- Faculty of Computer Science

