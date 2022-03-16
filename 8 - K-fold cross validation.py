"""
K - Fold Cross-Validation

    1. We would use the IRIS Dataset
    2. Can we create a model which can predict the iris Species
        given the following attributes:
        - Sepal Length, - Sepal Width
        - Petal Length, - Petal Width
        NOTE: We would use the SVC Model with Linear Kerne
    3. What if we use a K - Fold Cross-Validation with the
    above kernel?
    4. How about using a model with Polynomial Kernel?

"""

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC

"""
1. Load Dataset and define feature and target variables
"""


def load_iris_data_and_define_variables():
    global X
    global y

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # print(iris.DESCR)


load_iris_data_and_define_variables()

"""
2. Train-Test Splitting
"""


def train_test_split_iris():
    global X_train
    global X_test
    global y_train
    global y_test

    X_train, \
        X_test, \
        y_train, \
        y_test = train_test_split(X, y,
                                  test_size=0.4,
                                  random_state=0)


train_test_split_iris()

"""
3. Build model
"""


def build_svm_model(kernel):
    # this function makes it possible for you to
    # determine what kind of kernel you want your
    # model to be built with

    global model

    model = SVC(kernel=f"{kernel}",
                degree=3,
                C=1)   # build model

    model.fit(X_train, y_train)     # train model
    # print(model.score(X_test, y_test))


# build_svm_model('linear')   # Build model with LINEAR Kernel


"""
4. K-Fold Cross Validation

    It seems as if the 96% accuracy attained by the model
        is as a result of overfitting.
        
    Let us see if performing K-Fold Cross Validation would 
        change the results of the model.
        
    For this, we would be using the 'cross-val score' method
"""


def apply_k_fold_to_model(k):
    # we build our function in such a way that
    # we can determine how many folds ( k )
    # we want to use in our k-fold cross-validation
    global score

    score = cross_val_score(model, X, y, cv=int(f"{k}"))
    print(score)
    print(score.mean())


# apply_k_fold_to_model(5)

"""
5. Using a Polynomial Kernel

    So far, we have used only a 'linear' kernel
    Let us try a polynomial kernel this time
"""

# build_svm_model('poly')
# apply_k_fold_to_model(5)

# THERE WAS NO DIFFERENCE WITH THE LINEAR MODEL
# TO FIGHT OVER-FITTING, WE USED K-FOLD CROSS-VALIDATION
# POLYNOMIAL KERNEL DID NOT PROVIDE MUCH DIFFERENCE WITH RESPECT
# TO LINEAR KERNEL
