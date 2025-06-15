import pandas as pd
from sklearn.model_selection import train_test_split
from scripts.data_processing.feature_processing import preprocess_data
from scripts.models.logistic_regression import logistic_regression
from scripts.models.random_forest import random_forest
from scripts.models.xgboost import xgboost
from scripts.models.lgradboost import LGB

sep = "------------------------------"

def main():
    df = pd.read_csv('data/raw/Telco_Customer_Churn_Dataset.csv')
    print("Processing the Dataset")
    df_clean = preprocess_data(df)
    
    df_clean.to_csv('data/processed/cleaned_telco.csv', index=False)
    print(df['Churn'].value_counts(normalize=True))
    
    print(sep)
    print("Splitting the Model")

    X = df_clean.drop(columns=['Churn'])
    y = df_clean['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    
    print(sep)
    print("Training Logistic Regression Model")
    model = logistic_regression(df_clean)
    
    print(sep)
    print("Training Random Forest Model")
    model = random_forest(df_clean)

    print(sep)
    print("Training XGBoost Model")
    model = xgboost(df_clean)
    
    print(sep)
    print("Training Light Gradient Boost")
    model = LGB(df_clean)
    
    print(sep)
    return 0

if __name__ == "__main__":
    main()