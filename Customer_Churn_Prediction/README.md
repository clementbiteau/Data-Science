CUSTOMER CHURN PREDICTION

Predict whether a customer will churn based on demographics, account details, and service usage.
Predicting such event is critical for any business of any size or focus and reducing it is key to
grow in a competitive market.
Here, we will use the Telco Customer Churn Dataset to do a real world case study.

Tools:
    Python3
    Numpy
    Matplotlib
    Seaborn
    scikit-learn
    XGBoost
    imbalanced-learn
    SHAP
    VS Code

Objective:
    Our objective is to build a Machine Learning Model to predict churn and to identify the reasons behind 
    the clients' decision to turn away from using the product/services. This will enable us to better analyze
    which variables impact the churn.

Business impact:
    Knowing the reasons and the ways our customers churn will enable us to:
        - Take actions to retain high-risk churning customers
        - Improve customer experience
        - Innovate our offer and product
    
Source:
    [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
    Data contains:
        - Customer demographics (gender, senior citizen, etc.)
        - Account information (tenure, contract type, etc.)
        - Service usage (internet service, streaming services, etc.)
        - Churn label (Yes/No)


Evalutation Metrics: *Metrics are used due to potential class imalance*
    - F1 Score : Balance of Precision vs Recall - Imbalanced Datasets
    - ROC-AUC : Churn vs Not - Comparison of Models
    - Confusion Matrix :  - Debugging Models
    - Precision/Recall : N of lost churners / N of caught churners - Catching Risky Churners
    

STEPS
1. Data Exploration & Cleaning => eda : exploratory data analysis
    - Data fetching + cleaning
    - Charts + plots
    - Note important features
    - Imbalance Possibility
    
2. Feature Engineering
    - Transform data into meaningful features
    - Encoding Categorical Features
    - Scaling/Normalizing numeric features 

3. Model Building
4. Evaluation & Tuning
5. SHAP Interpretability
6. Business Insights