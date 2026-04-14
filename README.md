# 🎭 Emotion AI - Emotion Detection System

## 📌 Project Overview

This project is an AI-based Emotion Detection System that identifies human emotions using facial expressions through deep learning.

## 🚀 Features

* Real-time face detection using Haar Cascade
* Emotion classification using a pre-trained CNN (Convolutional Neural Network) model
* Supports multiple emotions like Happy, Sad, Angry, Neutral
* Provides visual and audio feedback based on detected emotion

## 🛠️ Technologies Used

* Python
* OpenCV
* TensorFlow / Keras
* NumPy

## 🧠 Model Details

The model is trained using a Convolutional Neural Network (CNN) to classify facial expressions into different emotion categories.

## 📸 Output

The system captures real-time video, detects faces, and displays the predicted emotion on the screen.

## 📂 Project Structure

* `app.py` → Main application file
* `emotion_model.h5` → Trained deep learning model
* `haarcascade_frontalface_default.xml` → Face detection model
* `Assets/` → Images and audio files

## ▶️ How to Run

1. Install dependencies:

   ```
   pip install opencv-python tensorflow numpy
   ```

2. Run the project:

   ```
   python app.py
   ```

## 🎯 Future Improvements

* Improve model accuracy
* Add more emotions
* Deploy as a web application

## 👩‍💻 Author

Gowthamee Gopinath
