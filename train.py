
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,roc_curve,confusion_matrix,classification_report)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import shap
import joblib


print('\nDataset Loading')

df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(df.head())

print(df.shape)

print(df.info())

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')

df['TotalCharges'].fillna(df['TotalCharges'].median(),inplace=True)

df.drop('customerID',axis=1,inplace=True)

df['AvgMonthlySpend'] = (df['TotalCharges'] / (df['tenure'] + 1))

df['TenureGroup'] = pd.cut(df['tenure'],bins=[0,12,24,48,72],
    labels=['0-1 Year','1-2 Years','2-4 Years','4-6 Years']
)

plt.figure(figsize=(6,4))
sns.countplot(x='Churn',data=df)
plt.title('Customer Churn Distribution')
plt.savefig('outputs/plots/churn_distribution.png')
plt.close()

plt.figure(figsize=(8,5))
sns.countplot(x='Contract',hue='Churn',data=df)
plt.title('Contract Type vs Churn')
plt.xticks(rotation=15)
plt.savefig('outputs/plots/contract_vs_churn.png')
plt.close()

plt.figure(figsize=(8,5))
sns.boxplot(x='Churn',y='MonthlyCharges',data=df)
plt.title('Monthly Charges vs Churn')
plt.savefig('outputs/plots/monthlycharges_vs_churn.png')
plt.close()

plt.figure(figsize=(6,4))
sns.countplot(x='gender',hue='Churn',data=df)
plt.title('Gender vs Churn')
plt.savefig('outputs/plots/gender_vs_churn.png')
plt.close()

plt.figure(figsize=(8,5))
sns.countplot(x='InternetService',hue='Churn',data=df)
plt.title('Internet Service vs Churn')
plt.savefig('outputs/plots/internetservice_vs_churn.png')
plt.close()

plt.figure(figsize=(8,5))
sns.countplot(x='PaymentMethod',hue='Churn',data=df)
plt.title('Payment Method vs Churn')
plt.xticks(rotation=25)
plt.savefig('outputs/plots/paymentmethod_vs_churn.png')
plt.close()

plt.figure(figsize=(8,5))
sns.boxplot(x='Churn',y='tenure',data=df)
plt.title('Tenure vs Churn')
plt.savefig('outputs/plots/tenure_vs_churn.png')
plt.close()

plt.figure(figsize=(8,5))
sns.histplot(df['TotalCharges'],bins=30,kde=True)
plt.title('Total Charges Distribution')
plt.savefig('outputs/plots/totalcharges_distribution.png')
plt.close()

plt.figure(figsize=(10,8))
correlation_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('outputs/plots/correlation_heatmap.png')
plt.close()

df['Churn'] = df['Churn'].map({'Yes':1,'No':0})

df = pd.get_dummies(df,drop_first=True)

X = df.drop('Churn',axis=1)

y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train,y_train)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_smote)

X_test_scaled = scaler.transform(X_test)

# LOGISTIC REGRESSION

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train_scaled,y_train_smote)

lr_predictions = lr_model.predict(X_test_scaled)

# RANDOM FOREST

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train_smote,y_train_smote)

rf_predictions = rf_model.predict(X_test)

# XGBOOST

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train_smote,y_train_smote)

xgb_predictions = xgb_model.predict(X_test)

# CATBOOST

cat_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    verbose=0
)

cat_model.fit(X_train_smote,y_train_smote)

cat_predictions = cat_model.predict(X_test)

# ANN MODEL

ann_model = Sequential()

ann_model.add(Dense(128,activation='relu',input_dim=X_train_scaled.shape[1]))

ann_model.add(Dropout(0.3))

ann_model.add(Dense(64,activation='relu'))

ann_model.add(Dropout(0.3))

ann_model.add(Dense(32,activation='relu'))

ann_model.add(Dropout(0.2))

ann_model.add(Dense(1,activation='sigmoid'))


ann_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


early_stop = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)


history = ann_model.fit(X_train_scaled,y_train_smote,validation_split=0.2,epochs=50,batch_size=32,callbacks=[early_stop],verbose=1)


ann_probability = ann_model.predict(X_test_scaled)

ann_predictions = (ann_probability > 0.5).astype(int).flatten()

# MODEL COMPARISON

models = {
    'Logistic Regression': (lr_predictions,lr_model.predict_proba(X_test_scaled)[:,1]),

    'Random Forest': (rf_predictions,rf_model.predict_proba(X_test)[:,1]),

    'XGBoost': (xgb_predictions,xgb_model.predict_proba(X_test)[:,1]),

    'CatBoost': (cat_predictions,cat_model.predict_proba(X_test)[:,1]),

    'ANN': (ann_predictions,ann_probability.flatten())
}


results = []

for name, (predictions, probabilities) in models.items():
    accuracy = accuracy_score(y_test,predictions)
    precision = precision_score(y_test,predictions)
    recall = recall_score(y_test,predictions)
    f1 = f1_score(y_test,predictions)
    roc_auc = roc_auc_score(y_test,probabilities)
    print(f'\n{name}')
    print('Accuracy:', round(accuracy,4))
    print('Precision:', round(precision,4))
    print('Recall:', round(recall,4))
    print('F1 Score:', round(f1,4))
    print('ROC-AUC:', round(roc_auc,4))
    results.append([name,accuracy,precision,recall,f1,roc_auc])

comparison = pd.DataFrame(results,
    columns=[
        'Model',
        'Accuracy',
        'Precision',
        'Recall',
        'F1 Score',
        'ROC-AUC'
    ]
)


comparison = comparison.sort_values(by='ROC-AUC',ascending=False)

print('\nModel Rankings')
print(comparison)

comparison.to_csv('outputs/reports/model_comparison.csv',index=False)

# MODEL COMPARISON PLOT

plt.figure(figsize=(12,6))
comparison_plot = comparison.melt(id_vars='Model',var_name='Metric',value_name='Score')
sns.barplot(data=comparison_plot,x='Model',y='Score',hue='Metric')
plt.title('Model Performance Comparison')
plt.xticks(rotation=15)
plt.ylim(0,1)
plt.savefig('outputs/plots/model_comparison.png')
plt.close()

# CONFUSION MATRIX

cm = confusion_matrix(y_test,xgb_predictions)
plt.figure(figsize=(6,5))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('outputs/plots/xgboost_confusion_matrix.png')
plt.close()

# CLASSIFICATION REPORT

print('\nClassification Report - XGBoost')
print(classification_report(y_test,xgb_predictions))

# FEATURE IMPORTANCE

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
})

feature_importance = feature_importance.sort_values(by='Importance',ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance',y='Feature',data=feature_importance.head(10))
plt.title('Top 10 Important Features')
plt.savefig('outputs/plots/feature_importance.png')
plt.close()

# ROC CURVE

xgb_probability = xgb_model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test,xgb_probability)
auc_score = roc_auc_score(y_test,xgb_probability)

plt.figure(figsize=(7,5))
plt.plot(fpr,tpr,label=f'AUC = {auc_score:.2f}')
plt.plot([0,1],[0,1],linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('outputs/plots/roc_curve.png')
plt.close()

# ANN TRAINING VISUALIZATION

plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'],label='Training Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.title('ANN Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('outputs/plots/ann_accuracy_curve.png')
plt.close()

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'],label='Training Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.title('ANN Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('outputs/plots/ann_loss_curve.png')
plt.close()

# SHAP EXPLAINABILITY

explainer = shap.TreeExplainer(xgb_model)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values,X_test,show=False)

plt.savefig('outputs/plots/shap_summary.png')

plt.close()

# CUSTOMER RISK SEGMENTATION

risk_probability = xgb_model.predict_proba(X_test)[:,1]

risk_df = pd.DataFrame()

risk_df['Probability'] = risk_probability

risk_df['RiskLevel'] = pd.cut(
    risk_df['Probability'],
    bins=[0,0.3,0.7,1],
    labels=[
        'Low Risk',
        'Medium Risk',
        'High Risk'
    ]
)

risk_df.to_csv('outputs/reports/risk_segmentation.csv',index=False)

# SAVE TRAINING COLUMNS

training_columns = X.columns.tolist()

joblib.dump(training_columns,'models/training_columns.pkl')

# SAVE MODELS

joblib.dump(xgb_model,'models/xgb_model.pkl')

joblib.dump(scaler,'models/scaler.pkl')

ann_model.save('models/ann_model.h5')
