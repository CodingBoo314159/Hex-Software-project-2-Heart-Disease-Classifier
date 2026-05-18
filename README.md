# Hex-Software-project-2-Heart-Disease-Classifier
Heart Disease Prediction System
Overview

This project is a machine learning application that predicts the likelihood of heart disease based on patient medical data. It uses a Random Forest model trained on structured health features and is deployed as an interactive web application using Gradio.

Features
Predicts heart disease risk using patient inputs
Outputs probability of risk
Provides feature importance for interpretability
Interactive web interface using Gradio
End-to-end machine learning pipeline
Dataset

The model uses a structured heart disease dataset containing features such as:

Age
Sex
Chest pain type
Blood pressure
Cholesterol
ECG results
Maximum heart rate
Exercise-induced angina
ST depression and slope
Model
Algorithm: Random Forest Classifier
Preprocessing: One-hot encoding for categorical variables
Pipeline: Scikit-learn Pipeline for preprocessing and training
Output: Binary classification with probability scores
Workflow
Load and clean dataset
Encode categorical features
Train Random Forest model
Evaluate using accuracy and feature importance
Save trained model
Deploy using Gradio interface
Deployment

The application is deployed using Gradio and allows users to input medical information to receive real-time predictions.

Requirements
Python
pandas
scikit-learn
gradio
joblib
