import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Loading the dataset
def load_data(path):
    df = pd.read_csv(path)
    print(f"Dataset Loaded Successfully")
    print(f"Dataset Shape: {df.shape}")
    
    return df

#Data Cleaning and Preprocessing
def clean_data(df):

    df = df.copy()

    df.drop_duplicates(inplace=True)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')

    numerical_columns = [
        'tenure',
        'MonthlyCharges',
        'TotalCharges'
    ]

    for col in numerical_columns:
        df[col] = df[col].fillna(df[col].median())

    categorical_columns = df.select_dtypes(include='object').columns

    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    if 'customerID' in df.columns:
        df.drop('customerID',axis=1,inplace=True)
        
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    
    df['Churn'] = df['Churn'].map({
    'No': 0,
    'Yes': 1
    })

    print("\nData Cleaning Completed")
    print(f"Final Dataset Shape: {df.shape}")
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDataset Info:")
    df.info()
    
    return df

#Encoding categorical features using one-hot encoding
def encode_features(df):
    df = pd.get_dummies(df,drop_first=True)

    print("\nEncoding Completed")
    print(f"Encoded Dataset Shape: {df.shape}")

    return df

# SPLIT FEATURES & TARGET
def split_features_target(df):

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    return X, y

# TRAIN TEST SPLIT
def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain-Test Split Completed")
    print(f"X_train Shape: {X_train.shape}")
    print(f"X_test Shape: {X_test.shape}")

    return (X_train,X_test,y_train,y_test)

# FEATURE SCALING
def scale_data(X_train, X_test):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("\nFeature Scaling Completed")

    return (X_train_scaled,X_test_scaled,scaler) 