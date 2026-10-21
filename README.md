# Customer Churn Prediction and Retention Analytics

An end-to-end, reproducible churn modelling project using the IBM Telco Customer Churn dataset. It cleans and audits the data, creates domain features, compares eight classifiers, evaluates them on one held-out test split, saves the selected pipeline, and exposes it through Streamlit.

## Models compared
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (RBF SVM)
5. K-Nearest Neighbours
6. XGBoost
7. CatBoost
8. Artificial Neural Network (implemented with scikit-learn `MLPClassifier`)

## Data cleaning and quality checks
- Strips leading/trailing whitespace from column names and text values.
- Converts `TotalCharges` to numeric; whitespace-only entries become missing values. The raw Telco CSV commonly has 11 such entries; `outputs/reports/data_quality_audit.json` records what is found in your copy.
- Checks and removes exact duplicate rows, recording the count. It does not remove rows merely because their feature values look similar.
- Removes rows with missing/invalid target labels; the target is never imputed.
- Drops `customerID`, which is an identifier rather than a predictive customer attribute.
- Leaves predictor imputation to the model pipeline: numerical columns use the training median, categorical columns use the training mode. One-hot encoding is fitted on training data and ignores unseen categories at inference.

## Feature engineering
- `AvgMonthlySpend`: `TotalCharges / tenure` when tenure is positive; uses current `MonthlyCharges` when tenure is zero to avoid division by zero.
- `TotalServices`: count of subscribed `Yes` values across OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, and StreamingMovies.
- `TenureGroup`: 0–12, 13–24, 25–48, and 49+ month categories.
- `MonthlyContractRisk`: marks month-to-month contracts. This is a descriptive feature association, not a causal claim.

These transformations are deterministic and shared by training and prediction. Learned imputation, encoding, and scaling are fitted only on the training portion.

## Evaluation methodology
- One stratified 60% train / 20% validation / 20% test split, random seed 42.
- All eight models use the same split and test rows.
- Scaling is included for Logistic Regression, SVM, KNN, and the neural network; tree-based models do not require scaling.
- Class imbalance is handled through class weighting or positive-class weighting for models that support it. No SMOTE is applied to one-hot encoded categorical data.
- A decision threshold is selected per model using **validation-set F2**. Test data is not used to tune that threshold.
- Reports include accuracy, balanced accuracy, precision, recall, F1, F2, ROC-AUC, confusion matrices, classification reports, and ROC curves.

### Which metric matters most?
Churn prevention usually values **recall** because missing a customer who will leave can mean lost recurring revenue. But high recall can generate many false alarms and costly retention offers. F2 is the primary comparison metric here because it weights recall more heavily than precision; ROC-AUC assesses ranking quality and precision helps contextualize intervention workload. In a real business, the best threshold should ultimately be selected using retention budget, intervention cost, and expected customer value. The script selects the winner using validation F2, with validation ROC-AUC as a secondary criterion; the test set is used only for the final comparison and reporting.

## Run in VS Code
Use Python 3.10 or 3.11 in a fresh virtual environment.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Put the raw CSV at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`. `train.py` rewrites generated reports, plots, and model artifacts on each run, so results from different runs are not silently mixed. Do not manually edit generated CSVs and treat them as model outputs.

## Generated outputs
- `outputs/reports/data_quality_audit.json`: duplicates, missing values, column types, target distribution.
- `outputs/reports/missing_values_after_cleaning.csv`, `dataset_summary.csv`, `split_summary.json`.
- `outputs/reports/model_comparison.csv`: comparable test metrics for all eight models.
- Per-model confusion matrices and classification reports.
- `outputs/reports/risk_segmentation.csv`: test-set predicted risk bands plus actual labels for evaluation only. This is not a future guarantee of churn.
- `outputs/plots/`: EDA and ROC curves.
- `models/best_model.joblib`, per-model pipelines, and `models/model_metadata.json`.

## Important limitations
- This is a retrospective benchmark on a public dataset, not proof that a retention action causes lower churn.
- Risk bands are illustrative probability ranges, not calibrated business policy.
- Model performance can change with library versions, random split, and data changes. Report only the results generated by your own final run.
- The app requires a full customer profile because the model was trained on multiple demographics, service, contract, billing, and payment attributes.
