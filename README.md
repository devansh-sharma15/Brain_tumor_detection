# Advance_Brain_Tumor_Classification
🧠 Brain Tumor Image Classification using CNN
This project is an advanced deep learning application focused on detecting brain tumors from MRI images using Convolutional Neural Networks (CNNs). It involves data preprocessing, model building, evaluation, and performance visualization.

📂 Project Overview
Brain tumors can be life-threatening and early detection is critical. This deep learning model classifies MRI brain images into tumor or no tumor categories using a custom CNN built with TensorFlow and Keras.

🚀 Features
Image data preprocessing with TensorFlow

CNN model architecture from scratch

Training with accuracy/loss visualization

Model evaluation with confusion matrix

Prediction on custom uploaded images

📁 Dataset
The dataset used contains MRI images categorized into:

yes/ – Images with tumor

no/ – Images without tumor

📦 Ensure your dataset is structured as:

Brain Tumor Data/

│

├── yes/

│   ├── image1.jpg

│   ├── ...

│

└── no/

    ├── image1.jpg
    
    ├── ...
    
🛠️ Tech Stack
Python

TensorFlow / Keras

NumPy

Matplotlib

Scikit-learn

📊 Model Architecture
Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense → Output

Activation: ReLU (hidden), Sigmoid (output)

Loss: Binary Crossentropy

Optimizer: Adam

📈 Results
Training Accuracy: ~99%

Validation Accuracy: ~95%

Loss Visualization: Tracked over epochs

Confusion Matrix: Shows true vs. predicted labels

📷 Prediction on Custom Image
The notebook includes functionality to upload and test custom MRI images for tumor prediction using the trained model.

▶️ Getting Started
Clone the repository:


git clone https://github.com/devansh-sharma15/Brain_tumor_detection.git
cd brain-tumor-classification
Install dependencies:


pip install -r requirements.txt
Run the notebook:
Open Advance DL Project Brain Tumor Image Classification.ipynb and execute cells step-by-step.

✅ To Do
Improve model with data augmentation

Experiment with pre-trained models (VGG16, ResNet)

Deploy using Streamlit or Flask

📄 License
This project is licensed under the MIT License.

🙌 Acknowledgements
Dataset by Kaggle Brain MRI Images for Brain Tumor Detection

TensorFlow & Keras documentation
