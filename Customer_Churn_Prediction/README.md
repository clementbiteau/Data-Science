CUSTOMER CHURN PREDICTION

Predict whether a customer will churn based on demographics, account details, and service usage.
Predicting such event is critical for any business of any size or focus and reducing it is key to
grow in a competitive market.
Here, we will use the Kaggle Telco Customer Churn Dataset to do a real world case study.

Tools:
    Python3
    Numpy
    Matplotlib
    Seaborn
    scikit-learn
    XGBoost
    LGBoost
    RandomForest
    CatBoost
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
    - Confusion Matrix :  - Debugging Models
    - Precision/Recall : N of lost churners / N of caught churners - Catching Risky Churners
    - Apply a top churner finding model to limit budget and means to recall
    

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
    - Split Data
    - Train Models
    - Evaluate Models
    - Tuning and Perfomance
    
    
$£¥€ Business Insights $£¥€

Finally, we have found the correct ML model that befits our needs.
The initial problem was "Work on the recall of the Telco dataset to limit customer churn".
We have done this by using a few different models including, Logistic Regression, Light Gradient Boost and Random Forest.
Whilst analyzing the Confusion Matrixes of said Models, we can clearly see that they were all quite strong in predicting the non churners.
Yet, the problem we are attempting to solve today, is the prediction of the churners.
In this, only the Light Gradient Boost was strong enough, with a 77% accuracy on recall to fill our needs.
Therefore, I have chosen this model to predict the churners.
Of course, we have to be aware that prediction is not an insurance of perfect targetting and thus, we may predict close to a quarter of churners when they actually are not at risk.

The question we can ask ourselves is: in a real world case study, is it not a safer prediction to suppose that all predicted churners are at risk ?

For real world purposes, I have added in the "/scripts/models/lgradboost.py" at lines 19-28 an additional precision.
We will assume that we do not have enough budget to send to the 100% predicted churners.
Here, we will suppose that we can only send marketing/comm offers to 80% of these predicted churners.
As the budget grows or diminishes, and as our theoretical case takes shape, we can simply change in the code the percentage of top churners at will.
Or, we can simply use the top_churners csv file to perform SQL manipulation, thus managing our needs versus our budget.