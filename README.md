# Customer Churn Prediction and Retention Analytics System

## Overview

This project focuses on predicting telecom customer churn using Machine Learning and Deep Learning techniques. The main objective of the project is to identify customers who are likely to leave the service and provide useful retention recommendations based on customer behavior.

The project includes:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Multiple model training and comparison
- Feature importance analysis
- Streamlit deployment

## Models Implemented

The following models were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost
- CatBoost
- Artificial Neural Network (ANN)

## Features

- Data Cleaning and Preprocessing
- Exploratory Data Analysis
- Feature Engineering
- SMOTE Balancing
- Multiple Model Comparison
- ROC Curve Evaluation
- Feature Importance Analysis
- Retention Recommendation System
- Interactive Streamlit Dashboard
- Real-Time Churn Prediction

## Dataset

IBM Telco Customer Churn Dataset

The dataset contains:
- Customer demographic details
- Subscription information
- Billing details
- Service usage information
- Churn status

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- XGBoost
- CatBoost
- Streamlit
- Matplotlib
- Seaborn
- Joblib

## Business Insights

Some important observations from the analysis:

- Customers with month-to-month contracts showed higher churn rates.
- Fiber optic users were more likely to churn.
- Electronic check payment method had a strong relation with churn behavior.
- Customers with long-term contracts were more likely to stay with the company.
- Higher monthly charges increased churn probability in many cases.

## Visualizations Included

The project includes several visualizations such as:

- Churn Distribution
- Contract vs Churn Analysis
- Monthly Charges vs Churn
- Correlation Heatmap
- ROC Curve
- Feature Importance Plot
- Model Comparison Graph
- Confusion Matrix

## Project Structure

```text
customer-churn-prediction/
│
├── app.py
├── train.py
├── requirements.txt
├── runtime.txt
├── README.md
├── .gitignore
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│
├── outputs/
│   ├── plots/
│   └── reports/
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── model_training.py
│   └── model_evaluation.py
```

## Running the Project

### Train the Models

```bash
python train.py
```

### Run the Streamlit Application

```bash
python -m streamlit run app.py
```

## Streamlit Application

The Streamlit application allows users to:
- Enter customer details
- Predict churn probability
- View retention recommendations
- Use the automatically selected best model

## Conclusion

This project demonstrates a complete end-to-end Machine Learning workflow for customer churn prediction. It combines predictive analytics, visualization, feature importance analysis, and deployment to provide meaningful business insights and customer retention strategies.