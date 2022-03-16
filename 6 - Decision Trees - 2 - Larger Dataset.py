"""
DECISION TREES - CONTINUED

Let Us now Use a Larger Dataset
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# load data

path = "file path/lending_club_new_data.csv"

loans = pd.read_csv(path)
# print(loans.head())
# print(loans.tail())      # to see how many rows we have
# print(loans.info())    # understand our data
# print(loans.describe())


"""
1. We create a 'good_loans' column using the 'bad_loans' 
    column.
    - Where we see a value of '0' in our 'bad_loans' column,
        we want to include a value of 'yes' in our 'good_loans' column.
    - Where we see a value of '1' in our 'bad_loans' column,
        we want to include a value of 'no' in our 'good_loans' column.
"""

loans['good_loans'] = loans['bad_loans'].apply(
    lambda yw: 'yes' if yw == 0 else 'no')

# print(loans.head())   # the new column has been added

"""
2. We define our features and our target variables.

    Our features involve every variable excluding the 'good_loans'
        and the 'bad_loans' variables.
    Our target variable is the 'good_loans' variable
"""

X = loans.drop(['bad_loans', 'good_loans'],
               axis=1)  # axis=1 means we are dropping columns
y = loans['good_loans']
# print(X.shape, y.shape)

"""
3. We use the train-test split method to split our dataset into 
    train and test subsets
"""

X_train, \
    X_test, \
    y_train, \
    y_test = train_test_split(X,
                              y,
                              test_size=0.3,
                              random_state=124)

"""
4. Build the model
"""


def build_model_cart():
    global predictions
    model = DecisionTreeClassifier()    # Instantiate model
    model.fit(X_train, y_train)     # Fit model
    predictions = model.predict(X_test)     # test model


build_model_cart()


"""
5. We evaluate the model
"""


def evaluate_model_cart():
    print('Confusion Matrix: ')
    print('')
    print(confusion_matrix(y_test,
                           predictions))    # Read very well on this
    print('')
    print('')
    print('')

    print('Classification Report: ')
    print('')
    print(classification_report(y_test, predictions))
    # we see that the model does very, very well

    # This is because we had more features, MORE DATA, and
    # so the learning algorithm learned the mapping function
    # very very well


# evaluate_model_cart()

"""
6. Let Us use the Random Forests classifier

6.1 we build our model
"""


def build_model_random_forest_classifier():
    global rf_predictions

    # Instantiate model
    rf_model = RandomForestClassifier(
        n_estimators=150
    )   # You can try out different values for n_estimators

    # Fit or train the model
    rf_model.fit(X_train, y_train)

    # Test the model
    rf_predictions = rf_model.predict(X_test)


build_model_random_forest_classifier()

"""
6.2 We evaluate the model
"""


def evaluate_model_random_forest_classification():
    print('Confusion Matrix (Random Forests): ')
    print('')
    print(confusion_matrix(y_test,
                           rf_predictions))  # Read very well on this
    print('')
    print('')
    print('')

    print('Classification Report (Random Forests): ')
    print('')
    print(classification_report(y_test,
                                rf_predictions))


evaluate_model_random_forest_classification()


"""
SUMMARY NOTES

1. Random Forest Classifiers (which are ENSEMBLES) generally
    do BETTER than Decision Tree Classifiers.
    
2. The MORE the DATA, the better the performance of the Model.

3. Try the classifications with LESS features and see what happens.

4. Use the technique above while defining features and target.
    a. To define features, drop the target column
    b. To define target, index the target column
    
    Simple 

"""
