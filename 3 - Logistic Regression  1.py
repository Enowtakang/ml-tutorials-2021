"""
LOGISTIC REGRESSION

Here, the output variable (y) or dependent variable has more than one outcome:
1. The first case is when it has 2 outcomes
    (Binary logistic regression) e.g. yes or no
2. The second case is when it has more than 2 outcomes
    (multinomial logistic regression) e.g. high, medium or low

LOGISTIC REGRESSION is preferably used then the input data (X) or independent
    variables contains both CATEGORICAL and NUMERICAL DATA.

The PURPOSE of logistic regression is to ANALYSE the effects of multiple variables
    (NUMERIC and/or CATEGORICAL) on the outcome variable

In LOGISTIC REGRESSION,
1. We predict the probability that the output belongs to each of the CLASSES
2. We DO NOT ASSUME A LINEAR RELATIONSHIP  between the independent variables and the
    dependent variable
3. We DO NOT ASSUME normality or linearity of the independent variables

LOGISTIC REGRESSION Uses Binomial Probability theory. READ MORE
    - Start with WIKIPEDIA

    -------------------------------------------------------------------

For today's analysis, we would use the following steps:

1. Exploring our data and visualization
2. Train and create a logistic regression model
3. Evaluate our model (Precision, F1 Score, Recall
4. What if we consider less features?
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

"""
1. Exploring our dataset and visualization
"""
# load dataset
path = "file path/binary.csv"
data = pd.read_csv(path)
# print(data.tail())      # to see how many rows we have
# print(data.info())    # understand our data
# print(data.describe())

# Visualize data
plt.figure(figsize=(10, 6))
plt.hist(data['gpa'], bins=35, color='blue') # histogram
plt.xlabel('GPA')
# plt.show()        # create histograms for the other columns

# we can see the relationship between GPA and GRE using a joint plot
# Research joint plots
sns.jointplot(x='gpa', y='gre',
              data=data, color='blue',
              kind='kde')   # what is kernel density estimation?
# plt.show()

# lets do the plot without 'kde'. Does it look better?
sns.jointplot(x='gpa', y='gre',
              data=data, color='blue')   # what is kernel density estimation?
# plt.show()                             # Please read the comment below
# it kind of looks more informative (when the 'kde' plot lacks pearson and p values)

"""
2. Train and create a logistic regression model

The 'rank' value is either 1, 2, 3, 4. This does not help much.

Let's create a DUMMY VARIABLE, to represent the data this way: 

        1, 2, 3, 4 = rank1, rank2, rank3, rank4
        
This way, we can convert each 'rank' value to either '0' or '1', as such:
    A rank value of 1 would be:
        1, 0, 0, 0
    A rank value of 2 would be:
        0, 1, 0, 0
    A rank value of 3 would be:
        0, 0, 1, 0
    A rank value of 4 would be:
        0, 0, 0, 1
"""
dummy_ranks = pd.get_dummies(data['rank'], prefix='rank')
# print(dummy_ranks.head())         # Great!!!


# Now that we have created the dummy variables, we have a risk: multi-collinearity.
# In multi-collinearity, one dummy variable e.g. rank_1 can have high correlation
# with another dummy variable e.g. rank_3; and this can happen MANY TIMES within
# the dummy variables.
# We avoid this by EXPLODING one variable e.g. rank_1


# The next step is to join our dummy variables to the
# admittance, GPA and GRE variables.
columns_we_need = ['admit', 'gre', 'gpa']   # we identify the columns which we need
new_data = data[columns_we_need].join(dummy_ranks.loc[:, 'rank_2':])

# dummy_ranks.loc[:, 'rank_2':] means that: we select all the rows from dummy_ranks
# (loc[:, ) then for the columns, we select starting from rank_2 till the end
# (, 'rank_2':] ).
# Thus, WE HAVE EXPLODED RANK 1 in order to avoid multi-collinearity.

# print(new_data.head())


# define dependent and independent variables
X = new_data[['gre', 'gpa', 'rank_2', 'rank_3', 'rank_4']]
y = new_data['admit']

# train-test splitting
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=23)
# print(X_train.shape)

# we create an instance of the Logistic Regression Class
model = LogisticRegression()

# fit the model
model.fit(X_train, y_train)

# model predict
prediction = model.predict(X_test)

# evaluate model (with classification report)
evaluation = classification_report(y_test, prediction)
# print(evaluation)     # read more about precision, recall, f1-score, support

# don't forget to do hyper-parameter optimization

"""
4. What if we consider less features?
    
    Would we get better results?
        Try this and see what happens. This is data science
"""
