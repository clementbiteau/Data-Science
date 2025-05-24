from xgboost import XGBClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


def xgboost(df):
    X = df.drop(columns=['Churn'], axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    
    
    model = XGBClassifier(use_label_encoder=False, eval_metrics='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["No Churn", "Churn"], 
                yticklabels=["No Churn", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - XGBoost")
    plt.tight_layout()
    plt.savefig("scripts/models/reports/XGB_confusion_matrix_rf.png")
    plt.close()
    
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    # Plot the classification report
    plt.figure(figsize=(8, 4))
    sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Classification Report - XGBoost")
    plt.tight_layout()
    plt.savefig("scripts/models/reports/XGB_classification_report_rf.png")
    plt.close()
    
    print(f"Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"Classification Report:\n", classification_report(y_test, y_pred))
    
    return model