"""
TEXT ANALYSIS

Here, we'll see:

    1. How to represent text in numerical format
    2. How to vectorize our dataset in document matrix
        (dtmatrix)
    3. Model Building and Evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer


"""
1. Let us take a look at the iris dataset
"""

# Load Iris dataset from sklearn
iris_data = datasets.load_iris()


# define features and labels
def define_features_and_target():
    global X
    global y
    X = iris_data.data
    y = iris_data.target
    # print(X.shape, y.shape)


define_features_and_target()

# let us create a pandas dataframe from our iris_data
iris = pd.DataFrame(X, columns=iris_data.feature_names)
# print(iris.head())


"""
2. Let's now go to text analysis
"""

# Consider the following training dataset
train_text = ['go home tonight',
              'go take a cab',
              'soccer! go play soccer...']

"""
    we need to convert our train_text into number format
    we need to make sure that each phrase has the same word count
    
    for this, we would use a count vectorizer
    
"""

# Let us instantiate a count vectorizer
vector = CountVectorizer()

# fit our Count Vectorizer with our training dataset
vector.fit(train_text)

# get feature names
gfn = vector.get_feature_names_out()
# print(gfn)

"""
when we get feature names, we discover three differences with 
    the original
    
    1. Get feature names gives the words in alphabetical order
    2. Has no word duplicates
    3. Has no punctuation
"""
