import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Encoding Target Churn with 1 Yes and 0 No
def encode_target(df):
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

# Encode Categorical Features
def encode_categoricals(df):
    df = df.drop(columns=['customerID'])
    df = pd.get_dummies(df, drop_first=True)
    return df

# handle numerical features and clean them
def handle_numericals(df):
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    df.reset_index(drop=True, inplace=True)
    return df

# pre process data to return a clean table
def preprocess_data(df):
    df = df.copy()
    df = handle_numericals(df)
    df = encode_target(df)
    df = encode_categoricals(df)
    return df