import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

MODEL_DIR = 'models'
PLOT_DIR = 'outputs/plots'
REPORT_DIR = 'outputs/reports'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# EVALUATION OF MACHINE LEARNING MODEL
def evaluate_model(model, X_test, y_test, model_name, feature_names=None):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test,y_pred,zero_division=0)
    recall = recall_score(y_test,y_pred,zero_division=0)
    f1 = f1_score(y_test,y_pred,zero_division=0)
    roc_auc = roc_auc_score(y_test,y_prob)

    print(f"\n{model_name} Evaluation")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    save_metrics(model_name,accuracy,precision,recall,f1,roc_auc)

    plot_confusion_matrix(y_test,y_pred,model_name)

    plot_roc_curve(y_test,y_prob,model_name)

    save_model(model,model_name)

    if feature_names is not None:
        plot_feature_importance(model,feature_names,model_name)

    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }

# EVALUATE ANN
def evaluate_ann(model, X_test, y_test):

    y_prob = model.predict(X_test)
    y_prob = y_prob.flatten()
    y_pred = (y_prob > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test,y_pred,zero_division=0)

    recall = recall_score(y_test,y_pred,zero_division=0)

    f1 = f1_score(y_test,y_pred,zero_division=0)

    roc_auc = roc_auc_score(y_test,y_prob)

    print("\nANN Evaluation")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    save_metrics('ANN',accuracy,precision,recall,f1,roc_auc)
    
    plot_confusion_matrix(y_test,y_pred,'ANN')

    plot_roc_curve(y_test,y_prob,'ANN')

    model.save(f'{MODEL_DIR}/ANN.h5')

    return {
        'Model': 'ANN',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }

# SAVE METRICS
def save_metrics(model_name,accuracy,precision,recall,f1,roc_auc):

    metrics_df = pd.DataFrame({
        'Metric': [
            'Accuracy',
            'Precision',
            'Recall',
            'F1-Score',
            'ROC-AUC'
        ],
        'Score': [
            accuracy,
            precision,
            recall,
            f1,
            roc_auc
        ]
    })

    metrics_df.to_csv(f'{REPORT_DIR}/{model_name}_metrics.csv',index=False)

# CONFUSION MATRIX
def plot_confusion_matrix(y_test,y_pred,model_name):

    cm = confusion_matrix(y_test,y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/{model_name}_confusion_matrix.png', bbox_inches='tight')
    plt.close()

# ROC CURVE
def plot_roc_curve(y_test,y_prob,model_name):

    fpr, tpr, _ = roc_curve(y_test,y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr,tpr,label=model_name)
    plt.plot([0, 1],[0, 1],linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/{model_name}_roc_curve.png', bbox_inches='tight')
    plt.close()

# FEATURE IMPORTANCE
def plot_feature_importance(model,feature_names,model_name):

    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_

    elif hasattr(model, 'coef_'):
        importance = model.coef_[0]

    else:
        return

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })

    importance_df['Importance'] = importance_df['Importance'].abs()

    importance_df = importance_df.sort_values(
        by='Importance',
        ascending=False
    ).head(15)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance',y='Feature',data=importance_df)
    plt.title(f'{model_name} Feature Importance')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/{model_name}_feature_importance.png', bbox_inches='tight')
    plt.close()

# MODEL COMPARISON CSV
def save_model_comparison(results):

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{REPORT_DIR}/model_comparison.csv',index=False)
    print("\nModel Comparison Report Saved")

# MODEL COMPARISON PLOT
def plot_model_comparison(results):

    results_df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model',y='F1-Score',data=results_df)
    plt.title('Model Comparison Based on F1-Score')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/model_comparison.png', bbox_inches='tight')
    plt.close()

# SAVE MODEL
def save_model(model,model_name):

    joblib.dump(model,f'{MODEL_DIR}/{model_name}.pkl')
    print(f"\n{model_name} Model Saved Successfully")

# SAVE BEST MODEL
def save_best_model(results, trained_models):

    results_df = pd.DataFrame(results)
    
    best_model_name = results_df.sort_values(
        by='F1-Score',
        ascending=False
    ).iloc[0]['Model']

    best_model = trained_models[best_model_name]

    if best_model_name == 'ANN':
        best_model.save(
            f'{MODEL_DIR}/best_ann_model.h5'
        )
    else:
        joblib.dump(
            best_model,
            f'{MODEL_DIR}/best_model.pkl'
        )

    print(f"\nBest Model: {best_model_name}")
    print("\nBest Model Saved Successfully")