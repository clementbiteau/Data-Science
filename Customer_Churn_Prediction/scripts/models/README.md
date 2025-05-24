MODELS - MACHINE LEARNING

The heart of our project is to set up three different models that will analyse a part of the data to be trained on, and then test the 
veracity of the predictions versus the test cases. The difference between the prediction and the true data is called "accuracy".
In this part, we will supervise three models of different algorithms and study the accuracy they offer on such data.

Prior working on models, we have cleaned and encoded our data to ensure safeguarding from bugs or any unexpected behaviours that could occur.
We have:
    - Deleted the rows with invalid data types (ie: TotalCharges)
    - Normalized the data (more efficient modelling)
    - Encoded categoricals with get_dummy()
These steps enable us to move to a safe Machine Learning environment.


TRAINING

- X axis the values of each categorical and numerical indexes EXCEPT ['Churn'];
- y axis the value of the target ['Churn'];

- X_train, y_train => used for 80% of our data; (base data to the model)
- X_test, y_test => used for 20% of our data; (new data to the model)

LOGISTIC REGRESSION

As we do not simply deal with numericals, and since we are attempting to find if a customer is "likely" to churn, we will start by using the essential Logistic Regression. Thus giving proper probability of churn in all of our customers.
In simple words, Logistic Regression transforms a Linear Regression model value with categorical value, using a sigmoid function.
It answers the question : "How likely is this customer a churning client ?". It is therefore not a numerical prediction, but a probability.

Sigmoid Function: P(y=1∣x)= 1 / (1 + e −(w⋅x+b))

    EVALUATION

    Confusion Matrix:
    [[916 117]
    [182 192]]

    Classification Report:
                precision    recall  f1-score   support
        0           0.83      0.89      0.86      1033
        1           0.62      0.51      0.56       374
    accuracy        0.79      1407
    macro avg       0.73      0.70    0.71    1407
    weighted avg    0.78      0.79    0.78    1407

    CONFUSION & CLASSIFICATION
                        Predicted
                    0               1
    Actual 0 [ True Negative - False Positive ]
    Actual 1 [ False Negative - True Positive ]

        Logistic Regression:
            Found that 916 clients did not churn correclty.
            Found that 117 did not churn wrongly.
            Found that 182 clients churned wrongly.
            Found that 192 clients churned correctly.

            Accuracy of finding the non churned customers: 100*(916)/(916+182) = 83%
            Accuracy of not missing the customers that did churn: 100*916/(916+117) = 89%
            F-1 score = Prediction vs Reality = (89% + 83%)/2 = 86%

            Accuracy of finding the churned customers : 100*192/(192+182) = 51%
            Accuracy of not mistakenly identify the churned customers : 100*(192)/(117+192) = 62%
            F-1 score = Prediction vs Reality => (51% + 62%)/2 = 56%

            Observation:
                Strong at identifying the customers that did not churn (83%)
                Weak at identifying the customers that actually churned (62%)
                Minimal error of prediction in the non churning customers => 86% correct.
                Impactful error of prediction in the churning customers => 56% correct only.

​	

RANDOM FOREST

A Random Forest is, conceptually, created from decision trees.
A Random Forest will be used for both classification and regression by making decision trees on multiple subsets taken from the data.
When the prediction time is reached, the forest aggregates the output of all the trees.

Random Forest will firstly classify each attribute individually and split the values to evaluate.

For instance, Monthly Charges are studied.
The decision trees will therefore study the thresholds from the data and ask the question "if i split the data here, how well does it separate the churners from non churners?".
The decision tree will then cease once it has found a threshold that separates effectively a maximum of churners vs non churners.
This is called fitting.
It uses a for of entropy or Gini Impurity algorithm.
Once it has found the threshold, it will effectively attribute that if a client's Monthly Charge is below the threshold then ... else ...

Thus creating an entirety of decision trees with thresholds and assumptions.

Note: the more tress does not mean the more accurate the Random Forest is. An alogrithm has its performance ceiling.

Therefore, the Random Forest could be seen as an ultimate judge, and the decision trees the jury.
When a new customer is studied, the jury will compare it to its many found thresholds and votes and will commit to a "vote".
This vote is what the Random Forest will return to the churning state of the current customer.

    EVALUATION
    Training Random Forest Model
    Confusion Matrix:
    [[917 116]
    [183 191]]
    Classification Report:
                precision    recall  f1-score   support

            0       0.83      0.89      0.86      1033
            1       0.62      0.51      0.56       374

    accuracy       0.79      1407
    macro avg       0.73      0.70      0.71      1407
    weighted avg       0.78      0.79      0.78      1407

    CONFUSION & CLASSIFICATION
                        Predicted
                    0               1
    Actual 0 [ True Negative - False Positive ]
    Actual 1 [ False Negative - True Positive ]

        Random Forest:
            Predictions:
                917 customers have been predicted to be non churners || on a total of 1100 non churners => 83%.
                191 customers have been prediced to be churners || on a total of 307 churners => 62%.
            Reality:
                89% of non churners have been identified.
                51% of churners have been identified.

        Observation:
                The Random Forest model was strong to detect the clients who will not churn.
                Though, it was weak in the churners identification by getting only 51% of the time the type
                of clients that would churn.
                The average accuracy might be comforting seeing an accuracy of 79%.
                But this is due to the great variance between the churners and non churners.
                Making our model biased. If we trust this current model blindly, we will miss half of our churners.


XGBOOST

    EVALUATION
    Confusion Matrix:
    [[903 130]
    [188 186]]
    Classification Report:
                precision    recall  f1-score   support

            0       0.83      0.87      0.85      1033
            1       0.59      0.50      0.54       374
    accuracy    0.77      1407
    macro avg       0.71      0.69      0.69      1407
    weighted avg       0.76      0.77      0.77      1407

        CONFUSION & CLASSIFICATION
                        Predicted
                    0               1
    Actual 0 [ True Negative - False Positive ]
    Actual 1 [ False Negative - True Positive ]

        XGBoost:
            Predictions:

            Observations:
                    The XGBoost model performs well at identifying customers who are likely to stay, with an 87% success rate in correctly classifying non-churners.
                    However, it struggles significantly to identify customers who are likely to churn — catching only 50% of them.
                    This means half of the at-risk customers are slipping through undetected, which is problematic for any business aiming to reduce churn through proactive retention strategies

