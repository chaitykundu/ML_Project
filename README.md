# Student Performance Prediction ML Pipeline

This repository implements an end-to-end **Machine Learning pipeline** for predicting student performance using the `stud.csv` dataset. It features modular components, proper logging, exception handling, and a simple web app for deployment.

# 🔑 Key Features

## 1. Data Ingestion
- Reads raw data (`stud.csv`) from the `notebook/` directory.  
- Splits data into training and testing sets.  
- Stores processed datasets under the `data/` directory.  

## 2. Data Transformation
- Handles preprocessing:
  - Missing values  
  - Categorical encoding  
  - Feature scaling  
- Outputs processed arrays ready for modeling.  

## 3. Model Training
- Trains multiple ML algorithms (CatBoost, Random Forest, etc.).  
- Selects the best-performing model.  
- Saves trained models and artifacts under the `artifacts/` directory.  

## 4. Web App Deployment (Flask)
- `app.py` + `templates/` provide a **Flask web interface**.  
- Users can input features via a form and receive predictions instantly.  

## 5. Experiment Tracking
- CatBoost training logs stored in `catboost_info/`.  

## 6. Error Handling & Logging
- Custom exception handling: `src/exception.py`.  
- Centralized logging for debugging and monitoring: `src/logger.py`.  
