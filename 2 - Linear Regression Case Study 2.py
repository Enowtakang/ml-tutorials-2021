"""
LINEAR REGRESSION CONTINUED

Here, we would:

1. Set up and visualize our data
2. Perform Univariate Linear Regression with NumPy
3. Perform Multivariate Linear Regression with sklearn

Research Question:

- Can we predict interest rates based on:
        1. a person's FICO (credit score)?
        2. a person's loan amounts?
"""

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import statsmodels.api as sm


# load data
path1 = "file path/loan.csv"
loans = pd.read_csv(path1)
# print(loans.head())
# print(loans.tail())
# print(loans.columns)

# let us check out the first 10 rows of the 'Loan.Length' column
loan_length_col_ten = loans['Loan.Length'][0:10]
# print(loan_length_col_ten)

"""
Univariate VISUALIZATION
"""
# 1. Box Plot of FICO Scores
plt.figure()
fico = loans['FICO.Score']  # Isolate the FICO Scores column
fico.hist(bins=20)
# plt.show() We see that the distribution is not normal

# 2. Box plot
plt.figure()
x = loans.boxplot('Interest.Rate', 'FICO.Score')
x.set_xlabel("Fico Score")              # x label
x.set_ylabel("Interest Rate in %")      # y label
# plt.show()  # as the FICO Score increases, the range of interest rate drops

"""
Multivariate VISUALIZATION
"""

# 1. Scatter plot matrix
pd.plotting.scatter_matrix(loans,
                           alpha=0.1,
                           grid=False,
                           color='red',
                           figsize=(10, 10),
                           diagonal='hist')   # alpha is the degree of transparency
# plt.show()

# From our scatter plot, we can see that the Loan Amount Had a linear relationship
# with both Fico Score and Interest rate.
# So we select Fico Score and Interest rate (feature selection) and then work with them.

"""
ANALYSIS  

We'll do a multi-variate linear regression, because we now know
that more than one variable affects our Loan Amount
"""

# Let's create three variables
interest_rate = loans['Interest.Rate']
loan_amount = loans['Loan.Amount']
fico_score = loans['FICO.Score']

# Now, we reshape our data from pandas dataframe to columns
y = np.matrix(interest_rate).transpose()    # y is our dependent variable
x_1 = np.matrix(fico_score).transpose()
x_2 = np.matrix(loan_amount).transpose()

# Now, we put our x_1 and x_2 together to form our input matrix
input_matrix = np.column_stack([x_1, x_2])

# Now, we create a linear model and fit it to our data
# we would base it on the Ordinary Least Squares method
x_3 = sm.add_constant(input_matrix)     # add our input matrix to stats models
model = sm.OLS(y, x_3)                  # we use OLS
model_fit = model.fit()                 # we fit our model

print("the p-values are: ", model_fit.pvalues)
print("the R-squared value is: ", model_fit.rsquared)

# If the p-value is <0.05, we have confidence that the
# independent variable in question really explains the dependent variable

# R0squared lies between -1 and 1
# R-squared is the coefficient of determination.
# It tells us how well the regression line approximates the data.
# Research more on line.

"""
CONCLUSION

We have a linear multi-variate regression model for interest rate
"""
