# 🐾 Animal classification using Deep Learning

A complete end-to-end pipeline to classify 10 different animals from images, built with PyTorch and deployed as a Streamlit web app. This repository includes:

- **`animal_prediction.ipynb`**: Jupyter notebook showing data loading, preprocessing, transfer learning with ResNet50, training, evaluation, and saving the model.  
- **`model_helper.py`**: All helper functions to load the trained model, preprocess an input image, and produce predictions (with confidence scores).  
- **`app.py`**: A Streamlit-based front-end for uploading an image and displaying the predicted animal with its confidence.  

---

## 🧐 Project Overview

This project trains a Convolutional Neural Network (CNN) to recognize the following 10 classes of animals:


Behind the scenes, it uses transfer learning on a pretrained ResNet50 backbone. After training, the final model is serialized to `saved_model.pth`, which `model_helper.py` loads at inference time. The Streamlit app (`app.py`) allows end users to upload any JPG/PNG/WEBP image and get a prediction in real time.

---

## 🗂️ Repository Structure

animal_model/
├── animal_prediction.ipynb # Notebook: Data preprocessing, model training & evaluation, saving saved_model.pth
├── model_helper.py # Python module: load_model, preprocess_image, predict (returns {prediction, confidence})
├── app.py # Streamlit app: upload image → display prediction
├── saved_model.pth # (Generated after running the notebook) Trained model weights
├── requirements.txt # All Python dependencies (see “Requirements” section)
└── README.md # ← You are here

animal_model/
└── dataset/
    ├── butterfly/
    │   ├── butterfly_01.jpg
    │   ├── butterfly_02.jpg
    │   └── …
    ├── cat/
    │   ├── cat_01.jpg
    │   ├── cat_02.jpg
    │   └── …
    └── … (and so on for all 10 classes)


dataset link:
https://www.kaggle.com/datasets/alessiocorrado99/animals10

