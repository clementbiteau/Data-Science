DATA PRE PROCESSING

This part is crucial to sort out raw data into a pre processed newly cleaned data.
It will take a few steps to ensure that our cleaned data befits our Machine Learning needs.
A Machine learning Algorithm needs to have numerical or binary data type. 1 and 0, True and False.

NUMERICALS

Firstly, we will handle any row that does not have a correct format in the numerical columns.
Here, we know from our info(), that TotalCharges should obvisouly be a float (numerical), yet some rows do not seem to cooperate with this assumption. This gives us objects.
We will therefore start by acting on this column and use to_numeric() function and coerce the errors.
This will ensure that all rows which were not numeric are now numeric.
Though, these rows will be dropped because they will most likely be empty.
We do not forget to reset the indexes in place as we dropped some rows.

TARGET

Secondly, we will transform our Churn Yes/No by 1 and 0.
We absolutely need to have this target as numerical and binary as it gets to properly study it as it is our subject.

CATEGORICALS

Lastly, we will cover the categoricals.
These columns are the non numerical columns that remain.
We dropped the ["customerID"] because it is purely factual and we cannot make any assumption from this dataset.
Here we purified our table again by dropping the customerID column altogether.
We will now use the method get_dummies() from pandas.
It will perform 2 important shifts:
    - Binary encoding
    - One-Hot Encoding

    BINARY ENCODING
        Binary encoding is transforming columns that have 2 options = Gender['Male', 'Female'].
        These columns will be replaced by a default ['male_yes'] and the result in rows will be True or False. Therefore 1 or 0.
        This ensures a proerly cleaned and efficient data that is now able to enter a ML Model safely.

    ONE-HOT ENCODING
        One-hot encoding takes the lead on columns (indexes) that require more than 2 choices.
        For instance our contract['month-to-month', 'one year', 'two year'] = 3 options.
        get_dummies() will One-hot encode this data into three different indexes =
        contract_one year = True || False
        contract_two_year = True || False
        If one is true than the other are false.
        If both are false then the customer is month-to-month.

    get_dummies() takes the data file and drops the first default created column to avoid collinearity.
    
    ie: if 3 columns are created, it is efficient to say that if 2 are false, then the third one is true. If one is true then the other 2 are false. => we can therefore delete a column.

    ie: if 2 columns are created, it is efficient to say that if 1 is false, then the other is true. therefore only one column is needed.

    time complexity improved with get_dummies(df, drop_first=true)