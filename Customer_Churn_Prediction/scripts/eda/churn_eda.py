import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load Data
df = pd.read_csv('../../data/raw/Telco_Customer_Churn_Dataset.csv')

# Inspect Data + Missing Values
print("Shape: ", df.shape)
    # print(df.head)
    # print(df.info())
    #print(df.isnull())
    # print(df.nunique())

# Clean TotalCharges index
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", pd.NA), errors='coerce')
df = df.dropna(subset=['TotalCharges'])
df.reset_index(drop=True, inplace=True)
print("Shape: ", df.shape)


# Plot Churn Distribution
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.savefig('../../data/plots/churn_distribution.png')
plt.close()
print(df['Churn'].value_counts(normalize=True))

# Churn vs Numeric Features
for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    sns.histplot(data=df, x=col, hue='Churn', kde=True, bins=30)
    plt.title(f"{col} by Churn")
    plt.savefig(f'../../data/plots/Numerical_Features/{col}_distribution.png')
    plt.close()

# Categorical Features
cat_cols = df.select_dtypes(include='object').columns.to_list()
cat_cols.remove('customerID')

for col in cat_cols:
    plt.figure(figsize=(6,3))
    sns.countplot(data=df, x=col, hue='Churn')
    plt.title(f"{col} vs Churn")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"../../data/plots/Object_Features/{col}_vs_churn.png")
    plt.close()

# Check Imbalance
y = df['Churn'].map({'Yes': 1, 'No': 0})
print(Counter(y))
