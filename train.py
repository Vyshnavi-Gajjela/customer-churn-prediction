"""Train and compare the eight churn models."""
from pathlib import Path
import json, shutil, warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay, roc_curve, auc
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import ChurnFeatureEngineer
from src.eda import generate_eda
from src.model_training import model_specs, build_pipeline
from src.model_evaluation import metric_row, select_threshold_for_f2

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT = ROOT / "outputs"
REPORTS = OUT / "reports"
PLOTS = OUT / "plots"
MODELS = ROOT / "models"
SEED = 42


def reset_generated_outputs():
    for folder in (REPORTS, PLOTS):
        folder.mkdir(parents=True, exist_ok=True)
        for file in folder.iterdir():
            if file.is_file(): file.unlink()
    MODELS.mkdir(parents=True, exist_ok=True)
    for file in MODELS.iterdir():
        if file.is_file() and file.suffix in {".joblib", ".json"}: file.unlink()


def main():
    reset_generated_outputs()
    df, audit = load_and_clean_data(DATA_PATH)
    REPORTS.mkdir(parents=True, exist_ok=True); PLOTS.mkdir(parents=True, exist_ok=True)
    with open(REPORTS / "data_quality_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    df.isna().sum().rename("missing_count").to_csv(REPORTS / "missing_values_after_cleaning.csv")
    df.describe(include="all").transpose().to_csv(REPORTS / "dataset_summary.csv")
    generate_eda(df, PLOTS)

    X = df.drop(columns="Churn").copy(); y = df["Churn"].copy()
    # 60% train / 20% validation / 20% final test, all stratified.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, stratify=y, random_state=SEED)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)
    split_summary = {"train_rows": len(X_train), "validation_rows": len(X_valid), "test_rows": len(X_test),
                     "train_churn_rate": float(y_train.mean()), "validation_churn_rate": float(y_valid.mean()), "test_churn_rate": float(y_test.mean()),
                     "random_state": SEED, "split": "60/20/20 stratified"}
    with open(REPORTS / "split_summary.json", "w", encoding="utf-8") as f: json.dump(split_summary, f, indent=2)

    rows = []; fitted = {}; validation_details = {}
    specs = model_specs(y_train)
    print(f"Dataset rows: {len(df)} | train={len(X_train)} validation={len(X_valid)} test={len(X_test)}")
    print("Training eight models. ANN is implemented as scikit-learn MLPClassifier.")
    for name, (estimator, scale_numeric) in specs.items():
        print(f"\nTraining {name}...")
        pipe = build_pipeline(estimator, X_train, scale_numeric)
        pipe.fit(X_train, y_train)
        valid_prob = pipe.predict_proba(X_valid)[:, 1]
        threshold, valid_f2 = select_threshold_for_f2(y_valid, valid_prob)
        test_prob = pipe.predict_proba(X_test)[:, 1]
        metrics = metric_row(y_test, test_prob, threshold)
        metrics.update({"model": name, "validation_f2_at_selected_threshold": valid_f2,
                        "validation_roc_auc": float(__import__('sklearn.metrics', fromlist=['roc_auc_score']).roc_auc_score(y_valid, valid_prob))})
        rows.append(metrics); fitted[name] = pipe
        validation_details[name] = {"threshold_selected_on_validation": threshold, "validation_f2": valid_f2}
        safe = name.lower().replace(" ", "_")
        joblib.dump(pipe, MODELS / f"{safe}.joblib")
        pred = (test_prob >= threshold).astype(int)
        pd.DataFrame(confusion_matrix(y_test, pred, labels=[0, 1]), index=["actual_no_churn", "actual_churn"], columns=["predicted_no_churn", "predicted_churn"]).to_csv(REPORTS / f"{safe}_confusion_matrix.csv")
        pd.DataFrame(classification_report(y_test, pred, labels=[0, 1], target_names=["No Churn", "Churn"], output_dict=True, zero_division=0)).transpose().to_csv(REPORTS / f"{safe}_classification_report.csv")
        fpr, tpr, _ = roc_curve(y_test, test_prob)
        fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}"); ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set(title=f"ROC curve — {name}", xlabel="False positive rate", ylabel="True positive rate"); ax.legend(loc="lower right"); fig.tight_layout()
        fig.savefig(PLOTS / f"{safe}_roc_curve.png", dpi=150); plt.close(fig)
        print(f"  validation-selected threshold={threshold:.2f}; test F2={metrics['f2']:.3f}; recall={metrics['recall']:.3f}; ROC-AUC={metrics['roc_auc']:.3f}")

    comparison = pd.DataFrame(rows)
    # Select the winning model using VALIDATION metrics only. The held-out test
    # metrics are reported for final assessment, never used to select the winner.
    comparison = comparison.sort_values(
        ["validation_f2_at_selected_threshold", "validation_roc_auc"],
        ascending=False,
    ).reset_index(drop=True)
    comparison.insert(0, "rank_by_validation_metrics", np.arange(1, len(comparison) + 1))
    comparison.to_csv(REPORTS / "model_comparison.csv", index=False, float_format="%.6f")
    best_name = comparison.iloc[0]["model"]
    best_pipe = fitted[best_name]
    joblib.dump(best_pipe, MODELS / "best_model.joblib")
    joblib.dump(best_pipe, MODELS / "churn_pipeline.joblib")

    # Test-set risk segmentation is descriptive, not ground-truth churn prediction.
    best_threshold = float(validation_details[best_name]["threshold_selected_on_validation"])
    best_prob = best_pipe.predict_proba(X_test)[:, 1]
    risk = pd.cut(best_prob, bins=[-0.000001, 0.30, 0.70, 1.000001], labels=["Low", "Medium", "High"], include_lowest=True, right=True)
    risk_df = pd.DataFrame({"predicted_churn_probability": best_prob, "predicted_risk_level": risk.astype(str),
                            "selected_classification_threshold": best_threshold, "actual_churn_for_evaluation_only": y_test.to_numpy()})
    risk_df.to_csv(REPORTS / "risk_segmentation.csv", index=False, float_format="%.6f")
    risk_df["predicted_risk_level"].value_counts().rename_axis("risk_level").to_frame("customers").to_csv(REPORTS / "risk_segmentation_summary.csv")

    # Feature names and model metadata for reproducibility.
    feature_names = list(ChurnFeatureEngineer().fit_transform(X_train).columns)
    metadata = {"best_model": best_name, "selection_rule": "Select highest validation-set F2 at a threshold tuned on validation data; use validation ROC-AUC and test precision as secondary reporting context. The test set is not used to choose the winning model.",
                "business_rationale": "Churn intervention often values finding at-risk customers, so F2 weights recall more than precision. Thresholds must ultimately reflect retention budget and intervention costs.",
                "thresholds_tuned_on_validation": validation_details, "feature_columns_before_encoding": feature_names,
                "target_mapping": {"No": 0, "Yes": 1}, "split": split_summary,
                "models_trained": list(specs.keys()), "note": "Test metrics are for final comparison/reporting; do not tune repeatedly against the test set."}
    with open(MODELS / "model_metadata.json", "w", encoding="utf-8") as f: json.dump(metadata, f, indent=2)
    with open(REPORTS / "run_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Best model selected using validation F2: {best_name}\n")
        f.write("Model ranking and operating thresholds are selected using validation data only.\n")
        f.write("Test metrics are for final evaluation and are not used to select the winner.\n\n")
        f.write(comparison.to_string(index=False))
    print("\n=== Model comparison (ranked by validation; test metrics shown) ===")
    print(comparison[["model", "validation_f2_at_selected_threshold", "threshold", "precision", "recall", "f1", "f2", "roc_auc", "accuracy"]].to_string(index=False))
    print(f"\nSelected model using validation F2: {best_name}")
    print(f"Reports: {REPORTS}\nPlots: {PLOTS}\nModels: {MODELS}")

if __name__ == "__main__": main()
