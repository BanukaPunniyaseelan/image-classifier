# CIFAR-10 Image Classifier

A convolutional neural network build with Tensorflow/Keras that classifies image into 10 categories
( airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck ), 
Trained on the CIFAR-10 dataset.

# **Live demo:** https://banukapunniyaseelan-image-classifier-app-xnf9wh.streamlit.app/

## Overview 
- Build and trained in Google Colab.
- Data augmentation (flip, rotation, translation) to reduce overfitting.
- 3-block CNN with batch normalization and dropout.
- Deployed as an interactive web app using Streamlit Community Cloud.

## Tech Stack
Python, Tensorflow/Keras, NumPy, Pillow, Streamlit

## How it works
- upload any photo - the model resizes it to 32x32 and returns its top 3 predicted categories with confidence scores.

## Files
- 'app.py' - Streamlit web app
- 'image_classifier.keras' - trained model
- 'requirements.txt' - dependencies
