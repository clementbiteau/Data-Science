import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def LGB(df):
    X = df.drop(columns=['Churn'], axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    
    model = lgb.LGBMClassifier(class_weight='balanced', num_leaves=31, learning_rate=0.05, n_estimators=100)
    model.fit(X_train, y_train)
    
    # Get 80% top predicted churners (for theoretical budget purposes)
    churn_probs = model.predict_proba(X_test)[:, 1]
    prob_df = X_test.copy()
    prob_df['Actual_Churn'] = y_test
    prob_df['Predicted_Prob_Churn'] = churn_probs

    prob_df_sorted = prob_df.sort_values(by='Predicted_Prob_Churn', ascending=False)

    top_80_percent = int(0.8 * len(prob_df_sorted))
    top_churn_risk_clients = prob_df_sorted.head(top_80_percent)
    top_churn_risk_clients.to_csv("data/top_churners/top_80_percent_churners.csv", index=False)

    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["No Churn", "Churn"], 
                yticklabels=["No Churn", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Light Gradient Boost")
    plt.tight_layout()
    plt.savefig("scripts/models/reports/LGB_confusion_matrix_rf.png")
    plt.close()
    
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    # Plot the classification report
    plt.figure(figsize=(8, 4))
    sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Classification Report - Light Gradient Boost")
    plt.tight_layout()
    plt.savefig("scripts/models/reports/LGB_classification_report_rf.png")
    plt.close()


    print(f"Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"Classification Report:\n", classification_report(y_test, y_pred))
    
    return model