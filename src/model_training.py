from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# HANDLE CLASS IMBALANCE
def handle_imbalance(X_train, y_train):
    print("\nBefore SMOTE:")
    print(y_train.value_counts())
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print("\nAfter SMOTE:")
    print(y_resampled.value_counts())

    return X_resampled, y_resampled

# CROSS VALIDATION
def perform_cross_validation(model, X_train, y_train):
    stratified_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=stratified_cv,
        scoring='f1'
    )   
    
    print("\nCross Validation F1 Scores:")
    print(scores)
    print(f"\nAverage F1 Score: {scores.mean():.4f}")

# LOGISTIC REGRESSION
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42)
    perform_cross_validation(model, X_train, y_train)
    model.fit(X_train, y_train)
    print("\nLogistic Regression Training Completed")

    return model

# DECISION TREE
def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(max_depth=6,class_weight='balanced',random_state=42)
    perform_cross_validation(model, X_train, y_train)
    model.fit(X_train, y_train)
    print("\nDecision Tree Training Completed")

    return model

# RANDOM FOREST WITH HYPERPARAMETER TUNING
def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(class_weight='balanced',random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_grid,
        n_iter=10,
        cv=3,
        scoring='f1',
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    perform_cross_validation(best_model,X_train,y_train)
    print("\nRandom Forest Training Completed")
    print("\nBest Parameters:")
    print(random_search.best_params_)

    return best_model

# SUPPORT VECTOR MACHINE
def train_svm(X_train, y_train):
    model = SVC(
        kernel='rbf',
        probability=True,
        class_weight='balanced',
        random_state=42
    )

    perform_cross_validation(model, X_train, y_train)
    model.fit(X_train, y_train)
    print("\nSVM Training Completed")

    return model

# K-NEAREST NEIGHBORS
def train_knn(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=7)
    perform_cross_validation(model, X_train, y_train)
    model.fit(X_train, y_train)
    print("\nKNN Training Completed")

    return model

# XGBOOST WITH HYPERPARAMETER TUNING
def train_xgboost(X_train, y_train):
    xgb_model = XGBClassifier(
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }

    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=10,
        cv=3,
        scoring='f1',
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    
    perform_cross_validation(
        best_model,
        X_train,
        y_train
    )

    print("\nXGBoost Training Completed")
    print("\nBest Parameters:")
    print(random_search.best_params_)
    return best_model

# CATBOOST
def train_catboost(X_train, y_train):

    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_state=42
    )

    perform_cross_validation(model, X_train, y_train)
    model.fit(X_train, y_train)
    print("\nCatBoost Training Completed")

    return model


# ARTIFICIAL NEURAL NETWORK
def train_ann(X_train, y_train, input_dim):
    
    model = Sequential()
    model.add(Dense(128,activation='relu',input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    print("\nANN Training Completed")

    return model