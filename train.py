import os
import joblib

from src.data_preprocessing import (
    load_data,
    clean_data,
    encode_features,
    split_features_target,
    split_data,
    scale_data
)

from src.feature_engineering import create_features

from src.eda import run_eda

from src.model_training import (
    handle_imbalance,
    train_logistic_regression,
    train_decision_tree,
    train_random_forest,
    train_svm,
    train_knn,
    train_xgboost,
    train_catboost,
    train_ann
)

from src.model_evaluation import (
    evaluate_model,
    evaluate_ann,
    save_model_comparison,
    plot_model_comparison,
    save_best_model
)

# LOAD DATA
df = load_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# CLEAN DATA
df = clean_data(df)

# FEATURE ENGINEERING
df = create_features(df)

# RUN EDA
run_eda(df)

# ENCODING
df = encode_features(df)

# SPLIT FEATURES & TARGET
X, y = split_features_target(df)

feature_names = X.columns
os.makedirs('models', exist_ok=True)
joblib.dump(feature_names,'models/feature_names.pkl')

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = split_data(X, y)

# FEATURE SCALING
X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

joblib.dump(scaler,'models/scaler.pkl')

# HANDLE CLASS IMBALANCE
X_train_resampled, y_train_resampled = handle_imbalance(X_train_scaled,y_train)

# RESULTS LIST
results = []

# TRAINED MODELS DICTIONARY
trained_models = {}

# LOGISTIC REGRESSION
lr_model = train_logistic_regression(X_train_resampled,y_train_resampled)
lr_result = evaluate_model(lr_model,X_test_scaled,y_test,'LogisticRegression',feature_names)
results.append(lr_result)
trained_models['LogisticRegression'] = lr_model

# DECISION TREE
dt_model = train_decision_tree(X_train_resampled,y_train_resampled)
dt_result = evaluate_model(dt_model,X_test_scaled,y_test,'DecisionTree',feature_names)
results.append(dt_result)
trained_models['DecisionTree'] = dt_model

# RANDOM FOREST
rf_model = train_random_forest(X_train_resampled,y_train_resampled)
rf_result = evaluate_model(rf_model,X_test_scaled,y_test,'RandomForest',feature_names)
results.append(rf_result)
trained_models['RandomForest'] = rf_model

# SVM
svm_model = train_svm(X_train_resampled,y_train_resampled)
svm_result = evaluate_model(svm_model,X_test_scaled,y_test,'SVM',feature_names)
results.append(svm_result)
trained_models['SVM'] = svm_model

# KNN
knn_model = train_knn(X_train_resampled,y_train_resampled)
knn_result = evaluate_model(knn_model,X_test_scaled,y_test,'KNN',feature_names)
results.append(knn_result)
trained_models['KNN'] = knn_model

# XGBOOST
xgb_model = train_xgboost(X_train_resampled,y_train_resampled)
xgb_result = evaluate_model(xgb_model,X_test_scaled,y_test,'XGBoost',feature_names)
results.append(xgb_result)
trained_models['XGBoost'] = xgb_model

# CATBOOST
cat_model = train_catboost(X_train_resampled,y_train_resampled)
cat_result = evaluate_model(cat_model,X_test_scaled,y_test,'CatBoost',feature_names)
results.append(cat_result)
trained_models['CatBoost'] = cat_model

# ANN
ann_model = train_ann(
    X_train_resampled,
    y_train_resampled,
    input_dim=X_train_resampled.shape[1]
)

ann_result = evaluate_ann(ann_model,X_test_scaled,y_test)
results.append(ann_result)
trained_models['ANN'] = ann_model

# SAVE MODEL COMPARISON
save_model_comparison(results)

# PLOT MODEL COMPARISON
plot_model_comparison(results)

# SAVE BEST MODEL
save_best_model(results,trained_models)

print("\nComplete Churn Prediction Pipeline Executed Successfully")