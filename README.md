# Customer Churn Prediction and Retention Analytics System

## Overview

This project focuses on predicting telecom customer churn using Machine Learning and Deep Learning techniques. The main objective of the project is to identify customers who are likely to leave the service and provide useful retention recommendations based on customer behavior.

The project includes:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training and comparison
- Explainable AI analysis
- Customer risk segmentation
- Streamlit deployment

## Models Implemented

The following models were trained and evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- CatBoost
- Artificial Neural Network (ANN)

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|-----------|-----------|---------|-----------|----------|
| ANN | 78.99% | 0.6219 | 0.5321 | 0.5735 | 0.8350 |
| Random Forest | 77.15% | 0.5563 | 0.6872 | 0.6148 | 0.8336 |
| CatBoost | 77.00% | 0.5592 | 0.6310 | 0.5930 | 0.8292 |
| XGBoost | 77.36% | 0.5703 | 0.5963 | 0.5830 | 0.8261 |
| Logistic Regression | 76.65% | 0.5556 | 0.6016 | 0.5777 | 0.8194 |

The ANN model achieved the highest accuracy and ROC-AUC score, while Random Forest produced the highest recall and F1-score among the Machine Learning models.

## Features

- Data Cleaning and Preprocessing
- Exploratory Data Analysis
- Feature Engineering
- SMOTE Balancing
- Multiple Model Comparison
- ROC Curve Evaluation
- SHAP Explainability
- Feature Importance Analysis
- Customer Risk Segmentation
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
- SHAP
- Streamlit
- Matplotlib
- Seaborn

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
- SHAP Summary Plot
- ANN Accuracy and Loss Curves
- Model Comparison Graph
- Confusion Matrix

## Project Structure

```text
customer-churn-prediction/
│
├── models/
├── outputs/
├── .gitignore
├── app.py
├── README.md
├── requirements.txt
└── train.py
```

## Running the Project

### Train the Models

```bash
python train.py
```

### Run the Streamlit Application

```bash
streamlit run app.py
```

## Streamlit Deployment

The Streamlit application allows users to:
- Enter customer details
- Predict churn probability
- Identify customer risk level
- View retention recommendations

## Conclusion

This project demonstrates a complete end-to-end Machine Learning workflow for customer churn prediction. It combines predictive analytics, explainable AI, visualization, and deployment to provide meaningful business insights and customer retention strategies.