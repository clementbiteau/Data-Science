import pandas as pd
from scripts.data_processing.feature_processing import preprocess_data

def main():
    df = pd.read_csv('data/raw/Telco_Customer_Churn_Dataset.csv')
    df_clean = preprocess_data(df)
    df_clean.to_csv('data/processed/cleaned_telco.csv', index=False)
    print(df['Churn'].value_counts(normalize=True))
    print("Shape: ", df_clean.shape)
    return 0

if __name__ == "__main__":
    main()