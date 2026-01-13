# Customer Churn Prediction Web Application

## Project Overview
Customer churn refers to customers leaving a service or stopping their subscription. Retaining existing customers is more cost-effective than acquiring new ones, making churn prediction a critical business problem.

This project is a **machine learning–based web application** that predicts whether a customer is likely to churn based on their demographic, service usage, and billing information. The application also uses **SHAP (Explainable AI)** to explain why the model made a particular prediction.

The system is built using **Python, Machine Learning, Flask, and SHAP**, and uses the **IBM Telco Customer Churn Dataset**.

## Problem Statement
Businesses face significant losses due to customer churn. Traditional rule-based methods fail to capture complex patterns in customer behavior.

This project aims to:
- Predict customer churn accurately using machine learning
- Provide transparency using explainable AI (SHAP)
- Deploy the solution as a user-friendly web application

## Features
- Machine Learning–based churn prediction
- Web interface built using Flask and HTML/CSS
- Uses IBM Telco Customer Churn dataset
- SHAP explanations for model predictions
- Scalable and modular project structure
- Suitable for real-world business use cases

## Dataset
**IBM Telco Customer Churn Dataset**

- Source: IBM Analytics
- Records: 7,000+ customers
- Features include:
  - Customer demographics
  - Account information
  - Services subscribed
  - Billing and payment details
- Target variable: `Churn` (Yes / No)

## Technologies Used
- Python
- Machine Learning (Scikit-learn)
- Flask (Web Framework)
- SHAP (Explainable AI)
- Pandas & NumPy
- HTML & CSS
- Pickle (Model persistence)

## Machine Learning Model
- Model Type: Random Forest Classifier
- Data preprocessing:
  - Encoding categorical variables
  - Feature scaling
- Model evaluation:
  - Accuracy
  - Precision
  - Recall
  - F1-Score

The trained model is saved and loaded using `pickle`.

## Explainable AI with SHAP
SHAP (SHapley Additive exPlanations) is used to explain individual predictions.

### Why SHAP?
- Shows how each feature impacts the prediction
- Improves trust and transparency
- Helps businesses understand churn drivers

### Use in this Project
- Explains why a customer is predicted to churn
- Highlights top contributing features
- Supports responsible and interpretable AI

## Application Workflow
1. User enters customer details through the web interface.
2. Input data is validated and preprocessed.
3. Categorical features are encoded and numerical features are scaled.
4. The trained machine learning model predicts whether the customer will churn.
5. SHAP explains the prediction by highlighting important features.
6. The prediction result is displayed on the web application.

## Use Cases
- Telecom companies to identify customers at risk of churn
- Subscription-based businesses to improve customer retention
- Marketing teams to target high-risk customers
- Data science learning and academic projects
- Demonstration of explainable AI in real-world systems

