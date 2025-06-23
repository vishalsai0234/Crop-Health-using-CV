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

You can download trained model from [here](https://huggingface.co/vishalsai0234/trained_model.keras/blob/main/trained_model.keras)
