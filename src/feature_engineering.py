import numpy as np
import pandas as pd

# FEATURE ENGINEERING
def create_features(df):
    df = df.copy()

    df['AvgMonthlySpend'] = (df['TotalCharges'] / (df['tenure'] + 1))

    service_columns = [
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies'
    ]

    df['TotalServices'] = 0

    for col in service_columns:
        df['TotalServices'] += (df[col] == 'Yes').astype(int)

    df['TenureGroup'] = pd.cut(df['tenure'],bins=[-1, 12, 24, 48, 72],
        labels=[
            'New Customer',
            'Regular Customer',
            'Loyal Customer',
            'Very Loyal Customer'
        ])

    df['HighValueCustomer'] = np.where(df['MonthlyCharges'] >df['MonthlyCharges'].median(),1,0)

    df['MonthlyContractRisk'] = np.where(df['Contract'] == 'Month-to-month',1,0)

    print("\nFeature Engineering Completed")

    print("\nNew Features Added:")

    new_features = [
        'AvgMonthlySpend',
        'TotalServices',
        'TenureGroup',
        'HighValueCustomer',
        'MonthlyContractRisk'
    ]

    for feature in new_features:
        print(f"- {feature}")

    print(f"\nDataset Shape After Feature Engineering: {df.shape}")

    return df