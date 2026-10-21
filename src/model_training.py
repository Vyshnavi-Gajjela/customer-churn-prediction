"""Model definitions and preprocessing pipeline."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.feature_engineering import ChurnFeatureEngineer

class CatBoostSklearnCompat(BaseEstimator, ClassifierMixin):
    """Small sklearn-compatible wrapper for CatBoost across sklearn versions."""
    def __init__(self, iterations=350, depth=6, learning_rate=0.04, class_weights=None, random_seed=42):
        self.iterations = iterations; self.depth = depth; self.learning_rate = learning_rate
        self.class_weights = class_weights; self.random_seed = random_seed
    def fit(self, X, y):
        self.model_ = CatBoostClassifier(iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
            loss_function="Logloss", eval_metric="AUC", class_weights=self.class_weights, verbose=False,
            random_seed=self.random_seed, allow_writing_files=False, thread_count=-1)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self
    def predict(self, X): return self.model_.predict(X)
    def predict_proba(self, X): return self.model_.predict_proba(X)


def build_preprocessor(X, scale_numeric):
    numeric = list(X.select_dtypes(include="number").columns)
    categorical = list(X.select_dtypes(exclude="number").columns)
    num_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale_numeric: num_steps.append(("scaler", StandardScaler()))
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([
        ("numeric", Pipeline(num_steps), numeric),
        ("categorical", cat_pipe, categorical),
    ], remainder="drop", verbose_feature_names_out=False)


def model_specs(y_train):
    positives = max(int((y_train == 1).sum()), 1); negatives = max(int((y_train == 0).sum()), 1)
    ratio = negatives / positives
    # class weights are a training-only response to the imbalanced target.
    return {
        "Logistic Regression": (LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42), True),
        "Decision Tree": (DecisionTreeClassifier(max_depth=7, min_samples_leaf=10, class_weight="balanced", random_state=42), False),
        "Random Forest": (RandomForestClassifier(n_estimators=350, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=42), False),
        "SVM": (SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=True, random_state=42), True),
        "KNN": (KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1), True),
        "XGBoost": (XGBClassifier(n_estimators=350, max_depth=4, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, min_child_weight=3, reg_lambda=2, scale_pos_weight=ratio, eval_metric="logloss", n_jobs=-1, random_state=42), False),
        "CatBoost": (CatBoostSklearnCompat(iterations=350, depth=6, learning_rate=0.04, class_weights=[1.0, ratio], random_seed=42), False),
        "Artificial Neural Network": (MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=0.001, batch_size=64, learning_rate_init=0.001, max_iter=250, early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, random_state=42), True),
    }


def build_pipeline(model, X_train, scale_numeric):
    return Pipeline([
        ("feature_engineering", ChurnFeatureEngineer()),
        ("preprocessor", build_preprocessor(ChurnFeatureEngineer().fit_transform(X_train), scale_numeric)),
        ("model", model),
    ])
