EXPLORATORY DATA ANALYSIS

This part will load, clean and shape up the raw data into a clear ordered dataset which we will plot in different manners to better
study the data. This will enable us to see clear trends and will be very effective to identify which situations cause the risk of churn.

LOAD

Firstly, we will load the data using pandas and store it.
We will undergo a short testing phase where we will:
    - print out the shape of the file (N of rows/cols);
    - print the indexes/titles
    - print the info of the data


CLEAN

We can spot an error there for 'TotalCharges'
Somehow, the data in TotalCharges is identified as an object when it should clearly be a Float (or a numeric value).
Indeed some clients have a " " in their TotalCharges, that shows when a new client has zero charges so far.
But this needs to be fixed.
Therefore, we will return the column of TotalCharges with putting to Numerical value all empty values to a NaN number.
We then finally drop the NaN missing values.
Our TotalCharges column is now cleaned and does not included the empty NaN rows anymore.
We reset the index by drop=True and inplace=True so that the data table is now cleaned and was amputated od 11 rows.

PLOT YES vs NO

Next step is to plot the distribution of the Churn.
In short, how many have churned vs how many have not churned.
We use countplot() from seaborn to create a line Yes and a line No. Which we save as a png file in our root of EDA.

=> We see that almost a third of our customers churn !
=> 26.6% Yes vs 73.4% No -> This is a very important churn rate.

PLOT NUM vs CHURN

Nex step is to compare the churn versus the numerical values in our dataset. tenure - MonthlyCharges - TotalCharges are all numerical (after we have cleaned TotalCharges).
We want to find which values have an impact on our churners. is it the spend ? the tenure (time they've been with us) ? other ?

Tenure => Clearly we see that new comers are much more likely to churn to "try and leave". Rather, our loyal long time customers practically never churn in comparison.

MonthlyCharges => From 20$ to 70$, our clients tend to stay with us. But when Monthly spending increases to >70$/month, our clientele seems to churn more.

TotalCharges => We can see an evident trend that clients that spend less are more likely to churn on a TotalCharges scale. The more they spend, the more they are loyal. That means we have an issue in promoting the first sales that would promote loyalty.


COUNTER

Finally, we will output a counter that will face the number of Yes versus the number of No in churning. Clearly, the imablance in classes is evident. We have 5,163 clients that are retained and 1,869 clients that churned.
We can therefore admit that the imbalance is such that our Machine Learning Model will have to be that more accute to take into account the churned customers as well as the potential churning customers. In our case, the "No" does not mean the clients will always be "No". Some might be on the brink of churning and those are the ones we must retain as soon as possible.


INTERPRETATION

Our data is clear. Our highest risk of churn happens in the early stages. New, low spending or high monthly spending customers are not properly attracted by our product.
We need to have an immediate and crucial action on onboarding, pricing and perhaps communication to hit early sales and retain customers after they have purchased. This will enable us to increase our number of loyal customers that tend to have a much higher Customer Lifetime Value (CLV).