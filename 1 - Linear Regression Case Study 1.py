"""
1. Exploring our data and visualization
"""

import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse


# GET DATA

path1 = "file path/home_data.csv"
house = pd.read_csv(path1)
# print(house.head())
# print(house.tail())
# print(house.info())
# print(house.describe())     # to obtain descriptive statistics
# print(house.columns)      # to see all columns in the dataset


# VISUALIZE DATA

# 1. Scatter Plot
# make a plt between square feet living (x-axis) and house price (y-axis) columns
# plt.scatter(house.sqft_living, house.price)     # x, y
# plt.xlabel("sqft of the house")     # label for x-axis
# plt.ylabel("price of the house")     # label for y-axis
# plt.show()      # show the plot

# 2. lmplot
# make a plt between square feet living (x-axis) and house price (y-axis) columns
# sns.lmplot("sqft_living", "price", data=house)
# plt.show()      # show the plot

# 3. Heatmap
# sns.heatmap(house.corr())
# plt.show()      # show the plot

# 4. distplot
# sns.distplot(house['price'], color='red')
# plt.show()      # show the plot

# 5. boxplot
# sns.boxplot(x='zipcode', y='price', data=house)
# plt.show()      # show the plot


"""
2. Performing Linear Regression
"""

# WE CHOOSE OUR FEATURES (or independent variables?)

# 1. We look at all our columns
columns = house.columns
# print(columns)

# 2. We choose our features
column_names_house = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                      'condition', 'sqft_above', 'yr_built', 'zipcode', 'sqft_lot15']

X = house[column_names_house]

# 3. we choose our label
y = house['price']

# TRAIN, TEST SPLITTING (we would use 70% for training and 30% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=7)
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)

# WE CREATE OUR MODEL

# Instantiate the model
model = LinearRegression()

# fit the model
model.fit(X_train, y_train)

# WE EVALUATE THE MODEL

# before testing the model, we need to use it to predict the results of X_test data
# please note that the predictions are all stored in the 'predictions' variable below
predictions = model.predict(X_test)

# note: you can retrieve specific rows of data from your list, then test the model
# predictions on it.
# Example: the id number on index number '1' is 6414100192.
# We would observe that the actual price of the house is 538000 but the model
# gave a prediction of about 724005
house_id_number = 6414100192
house_id = house['id'] == house_id_number
house_1 = house[house_id]
# print(house_1)
# print(house_1['price'])
# print(predictions[1])

# Let us present PREDICTION vs ACTUAL PRICES on a scatter plot
plt.figure(figsize=(10, 5)) # determine the size of your figure (length, width)
plt.scatter(y_test, predictions)    # x, y
# plt.show()

# to find out the coefficient values (or WEIGHTS) for our model features, we call
# the 'coef_' method
model_coefficients = model.coef_
# print(model_coefficients)

# we can then map each coefficient (WEIGHT) to the specific feature to see what effect
# each feature has on the price
map_coefficients = pd.DataFrame(model_coefficients,
                                X.columns,
                                columns=['Coefficient Values']) # we name the columns
# print(map_coefficients)
# we interpret the results as follows: if the weight for bathrooms is 10 (for example),
# it means that a unit increase in bathrooms would result in a 10 dollars increase in
# house price.
# HERE, WE START TALKING ABOUT FEATURE IMPORTANCE
# sometimes this data may not be realistic. WE NEED TO TUNE THE HYPER-PARAMETERS.

# 'model.intercept_' tells us at what point the regression line CROSSES the y-axis
intercept_model = model.intercept_
# print(intercept_model)

"""
3. Model Evaluation
"""

# MODEL PERFORMANCE METRICS

# 1. We start with Root Mean Square Error (RMSE).
# We expect a model to have as small an RMSE error as possible.
# Procedure. We calculate the mean squared error (mse below) with sklearn
# We then square root the results to get our RMSE
mSe = mse(y_test, predictions)
# print(mSe)
r_mSe = math.sqrt(mSe)
# print(r_mSe)

# We can also use numpy (np) to calculate our RMSE
np_r_mSe = np.sqrt(mSe)
# print(np_r_mSe)

"""
4. We saw that our RMSE was high.
    - We can then choose different features and then calculate our RMSE from a new 
        model which we make with these different features.
    - We can repeat this with several feature combination until we get a minimum
        RMSE value
        
    - ****
        The lesson here is that: YOU MUST NOT USE ALL OF THE DATA
      ****
    
    - ****
        WE NEED TO USE THE --MOST LOGICAL-- FEATURES
      ****
     
"""
