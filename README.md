# Customer Churn Prediction Web Application

## Project Overview
Customer churn refers to customers leaving or discontinuing a service. Predicting churn in advance helps businesses take proactive steps to retain customers.

This project is a **Machine Learning–based web application** that predicts whether a customer is likely to churn based on demographic, service usage, and billing information. The system also uses **SHAP (Explainable AI)** to explain why a particular prediction was made.

The application is built using **Python, Machine Learning, Flask, and SHAP**, with a simple HTML/CSS frontend for user interaction.

## Problem Statement
Customer retention is a major challenge for subscription-based businesses. Traditional rule-based approaches are insufficient to capture complex customer behavior.

This project aims to:
- Predict customer churn accurately using machine learning
- Provide transparent and interpretable predictions using SHAP
- Deploy the solution as a real-world usable web application

## Dataset
**IBM Telco Customer Churn Dataset**

- Provided by IBM Analytics
- Contains customer demographic and service-related information
- Target variable: `Churn` (Yes / No)
- Includes features such as:
  - Tenure
  - Monthly charges
  - Contract type
  - Internet services
  - Payment method

## Technology Stack
- Python – Core programming language
- Flask – Backend web framework
- Scikit-learn – Machine learning algorithms
- SHAP – Explainable AI
- Pandas & NumPy – Data preprocessing
- HTML & CSS – Frontend user interface
- Pickle – Model persistence

## Machine Learning Model
- Model used: Random Forest Classifier
- Data preprocessing:
  - Categorical feature encoding
  - Feature scaling
- Model evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score

The trained model and preprocessing objects are saved and reused to ensure consistent predictions.

## Explainable AI with SHAP
SHAP (SHapley Additive exPlanations) is used to interpret model predictions.

### Purpose of SHAP in this Project
- Explains how each feature contributes to churn prediction
- Improves transparency and trust in the model
- Helps businesses understand key churn drivers

## Application Workflow
1. User enters customer details through the web interface  
2. Input data is validated and preprocessed  
3. Features are encoded and scaled using saved transformers  
4. Trained machine learning model predicts churn probability  
5. SHAP explains the prediction by highlighting feature impact  
6. Final prediction is displayed to the user  

## Use Cases
- Telecom companies predicting customer churn
- Subscription-based services improving retention strategies
- Customer success teams identifying high-risk customers
- Data science students learning ML deployment
- Business analysts understanding churn behavior


