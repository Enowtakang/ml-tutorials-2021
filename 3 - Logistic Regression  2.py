"""
LOGISTIC REGRESSION CASE STUDY ON PIMA INDIANS DATASET

Here, we intend to predict, given some parameters,
    weather a person is diabetic or not.
    This BINARY outcome requires logistic regression.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# load dataset
path = "file path/pima.csv"

data = pd.read_csv(path)
# print(data.head())
# print(data.tail())      # to see how many rows we have
# print(data.info())    # understand our data
# print(data.describe())

"""
When we ran the <<print(data.head())>> command, 
    we realised that the first row of data was used as the COLUMN NAMES.
    We need to do something about it.
"""

# we create a list of column names which we would give to the columns
column_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin_level',
                'bmi', 'pedigree', 'age', 'diabetes_label']

# we then reread the data, adding our columns as headers
diabetes_data = pd.read_csv(path, names=column_names)
# print(diabetes_data.head())
# print(diabetes_data.count())    # Each column has 768 data units

"""
We identify features, input and output values
"""

features = ['pregnant', 'insulin_level', 'bmi', 'age']
X = diabetes_data[features]
y = diabetes_data.diabetes_label    # Another way to index a column

# TRAIN, TEST SPLITTING (we would use 70% for training and 30% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=0)

# create and fit model
model = LogisticRegression()
model.fit(X_train, y_train)

# make predictions in order to test the model
y_pred = model.predict(X_test)

# Evaluate the model using its proper metrics
# print(metrics.accuracy_score(y_test, y_pred))

"""
We further evaluate our model
"""

# First, we get the frequency of each distinct value in the 'diabetes_label' column
v_c = y_test.value_counts()
# print(v_c)

# Print all the values for y_test
# print("True diabetes label: ", y_test.values)

# Print the first 50 values for y_test
# print("     True diabetes label: ", y_test.values[0:10])

# Print the first 50 values for y_pred
# print("Predicted diabetes label: ", y_pred[0:10])

# Print confusion_matrix (READ ON CONFUSION MATRIX
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
# print(confusion_matrix)   # read wikipedia's "sensitivity and specificity" article

"""
Model Error
(FP + FN) / (TP + TN + FP + FN)
FP = False Positive(s)
TN = True Negative(s)

Sensitivity
Read from wikipedia
"""

# Sensitivity
sensitivity = metrics.recall_score(y_test, y_pred)
# print(sensitivity)

# precision_score
p_s = metrics.precision_score(y_test, y_pred)
# print(p_s)
