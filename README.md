# 🌿 Crop Health using Computer Vision

This project uses deep learning (specifically CNNs) to detect plant diseases from images of leaves. The idea is to help farmers — especially in areas with limited access to agricultural experts — identify plant diseases early using just a photo.

---

## 🚀 What This Project Does

- Classifies leaf images into healthy or diseased categories
- Uses a Convolutional Neural Network (CNN) built with TensorFlow
- Trained on the **New Plant Diseases Dataset (Augmented)** from Kaggle
- Evaluated using metrics like accuracy, precision, recall, and F1 score
- Aims to support real-world use via mobile or web-based platforms

---

## 🧠 Tech Stack

- Python 🐍  
- TensorFlow & Keras  
- Matplotlib & Seaborn (for visualization)  
- Kaggle for the dataset  

---

## 📁 Dataset

The dataset includes images of plant leaves — both healthy and affected by various diseases — organized into folders per class.

**Source:**  
[Kaggle - New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

- ~38 classes (including multiple diseases per crop)  
- Pre-augmented and resized to 128x128 RGB  
- One-hot encoded labels for multi-class classification
- Note: Make sure to download the dataset from Kaggle and place it in the **data/** directory.

---

## 📊 Results

The CNN model achieved strong performance on the validation set, with most misclassifications occurring between visually similar diseases. The results suggest good potential for field use, especially if wrapped in a mobile app for farmers.

---

## 📂 Directory Structur:
├── data/                            # Sample test images  
├── model/                           # Training history + downloadable trained model (from [here](https://huggingface.co/vishalsai0234/trained_model.keras/blob/main/trained_model.keras))

├── notebooks/                       # Jupyter notebooks for experimentation and analysis  
├── src/                             # Image and prediction script for web interface  
├── Crop_Health_using_CV_Report.pdf  # Project report  
├── README.md                        # Project overview and usage  
└── requirements.txt                 # Dependencies and library versions

---

## ⚙️ Command:

1. Open Jupyter Notebook or Google Colab and run the notebook files sequentially.

2. After running "2_Model_Building_&_Training.ipynb" file successfully, a new file called "trained_model.keras" is created in the folder.

3. Now, use this "trained_model.keras" file in subsequent notebooks and the Python script

4. Then open a terminal from the folder which contains this files

5. Run the Streamlit app using the following command to predict the type of leaf image with disease:
```bash
streamlit run main.py
``` 

7. If you getting error then proceed with using following commands in terminal (to create new environment):
```bash
conda create -n tf_env python=3.10
conda activate tf_env
conda install tensorflow pandas numpy matplotlib scikit-learn
streamlit run main.py
```

Now, terminal automatically open web browser for streaming.
