"""
OUTLIERS

    1. Sometimes, outliers can skew our results to the
        right or to the left.
    2. Usually, we use standard deviation as a way of finding
        outliers.
    3. Usually, if a value is within 1 to 2 times the standard
        deviation, it os Okay.
        Out of this, it should be considered an outlier.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
1. CASE STUDY

    We have data for the yearly income in a country, 
        say NIGERIA
    Let us create a normal distribution EXAMPLE for 
        that income data.
        We give the mean income as 30,000 Naira and a standard
            deviation of 10,000 for a group of 20,000 people.
"""
income_nd = np.random.normal(30000,
                             10000,
                             20000)     # mean, std, pop
"""
2. We plot the normal distribution using a histogram plot.
"""


def plot_nd_with_histogram(data, n_o_b):
    # n_o_b means 'number of beans'
    plt.hist(data, n_o_b)
    plt.show()


# plot_nd_with_histogram(income_nd, 25)

"""
3. From the above plot, we have seen how balanced our 
    normal distribution is.
    
    Let us introduce an outlier to our data, and see how 
        it affects our distribution.
        This outlier should preferably be a higher income, 
            say 2bn (for the purposes our example) 
"""
# we add the 2bn outlier value
income_nd = np.append(income_nd,
                      [2000000000])
# we plot the distribution once again
# plot_nd_with_histogram(income_nd, 25)

"""
4. From the plot in three (3.) above, we see that the 
    distribution is unrecognisable with respect to the 
        good plot in two (2.) above.
        
    - So we define an acceptable value as a value which is 
        1 to 2 times the standard deviation.
    -  We  define an outlier as a value which is >2 times the 
        standard deviation.
        
    Let us define a function which would remove outliers:
        It would do this by SELECTING ONLY values which appear
            in the normal range.
"""


def remove_outliers(data):
    # We identify the mean and the median values for the data
    # Please note that the 'data' here is the list of values
    # in a SINGLE variable.

    median = np.median(data)
    sd = np.std(data)

    # We then use list comprehension to isolate
    # the outliers in the data.
    # Note that 'n' below stands for 'new_values'.

    global n
    n = [x for x in data if (median-1.5*sd < x < median + 1.5*sd)]
    return n


# we then plot back our histogram after removing the outliers
# from our 'income_nd' data
n = remove_outliers(income_nd)

# plot_nd_with_histogram(n, 25)

"""
 We see from the plot above (in four (4)) that the histogram 
    is back to normal. 
    
    - Our 2bn outlier was effectively removed.
    
    - We can therefore use this function to remove outliers.
"""
