"""
CLUSTER ANALYSIS

- This is an unsupervised learning method.
- The GOAL is to GROUP or CLUSTER observations into subsets
    based on the SIMILARITY of responses on MULTIPLE VARIABLES
- Observations with similar response patterns are grouped together
- EXAMPLE
    Researchers might use cluster analysis to IDENTIFY individuals
    that are at GREATEST RISK for health problems, and develop TARGETED
    health messages based on patterns of health behaviour

- With cluster analysis - we want to obtain clusters which have 'less variance'
    'within clusters' and 'more variance' 'between clusters'.
- More variance between clusters means that THE CLUSTERS ARE UNIQUE

- We want observations within clusters to be similar to each other than
    they are to observations in other clusters

- Check out the sklearn-documentation for more explanation and examples
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

"""
1. LOAD THE DATASET
----------------------------------

It is important to note the meanings of the cluster variables:

Alcprobs1   = Alcohol Problems. This variable has a range from 1 - 6
Deviant1    = Deviant Behaviour
Viol        = Violent Behaviour Scale
dep         = Depression
paractv     = Parental Activity
parpres     = Parental Presence
famconct    = Family Connectedness
esteem      = Self Esteem

"""

path = "file path/health_data.csv"

data = pd.read_csv(path)
# print(data.head())
# print(data.tail())      # to see how many rows we have
# print(data.info())    # understand our data
# print(data.describe())

"""
2. CLEAN THE DATASET
----------------------------------

"""

# Let us now change all 'column names' in the data to UPPERCASE
data.columns = map(str.upper, data.columns)

# Next, we remove all 'NaN' values
data = data.dropna()
# print(data.head())

"""
3. SELECT RELEVANT FEATURES
----------------------------------

We would ONLY consider the following clustering variables for our job
    Always remember that you have POWER over the data
    You manipulate the data as you deem necessary
"""
lst = ['ALCEVR1', 'MAREVER1', 'ALCPROBS1',
       'DEVIANT1', 'VIOL1', 'DEP1',
       'ESTEEM1', 'SCHCONN1', 'PARACTV',
       'PARPRES', 'FAMCONCT']

clustering_variables = data[lst]

"""
4. STANDARDIZE OUR DATASET
----------------------------------

Since that our clustering may be influenced by variables which are measured
on larger scales, we need to STANDARDIZE the variables so that they all have 
a MEAN of ZERO (0) and a STANDARD DEVIATION OF ONE (1)
"""

# 4.1 we first make a copy of our clustering variables
copy = clustering_variables.copy()

"""
4.2 Now, we standardize each FEATURE which we selected in step 3

The steps below help with the task:
"""
# First, we make a list of the variables which we  want to standardize:
list_of_vars_for_std = ['ALCEVR1', 'MAREVER1', 'ALCPROBS1', 'DEVIANT1',
                         'VIOL1', 'DEP1', 'ESTEEM1', 'SCHCONN1', 'PARACTV',
                         'PARPRES', 'FAMCONCT']


# Then, we create a function to standardize each variable:
def standardize_variables(list):
    # We standardize each variable in our list of variables
    for variable in list:
        copy[variable] = preprocessing.scale(copy[variable].astype('float64'))


# We now call our function and pass our list, in order to standardize
# our variables
standardize_variables(list_of_vars_for_std)


"""
5. Let us do K-Means cluster analysis for 1 to 10 clusters
----------------------------

"""

# 5.1. we start by splitting our dataset into training and testing subsets
cluster_train, cluster_test = train_test_split(copy,
                                               test_size=0.3,
                                               random_state=222)

# print(cluster_train.shape, cluster_test.shape)


# 5.2. Then, we define some variables
clusters = range(1, 11)     # From 1 to 11, so that numbers from 1 to 10 are included
mean_dist = []              # Empty list for now


# 5.3. We build our K-Means model
for K in clusters:
    # we create model and set the number of clusters to K
    model = KMeans(n_clusters=K)

    # We use the training dataset to fit our model
    model.fit(cluster_train)

    # We now calculate our euclidean distances using the 'cdist'
    # function from scipy and stock the values in the 'mean_dist'
    # variable in step 5.2
    mean_dist.append(sum(np.min(cdist(cluster_train,
                                      model.cluster_centers_,
                                      'euclidean'),
                                axis=1))/cluster_train.shape[0])


"""
6. We plot the elbow curve
-------------------------------
"""
# we define a function and store the code there


def elbow_curve():
    plt.plot(clusters, mean_dist)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average distance')
    plt.title('Elbow curve for our K values')
    plt.show()


# elbow_curve()


"""
From the curve, we see that as the number of clusters increase,
    the average distance between clusters drops

Also, we see that the elbow curve BENDS at 2, 3, and 6 clusters.
    Deciding which number of clusters to choose may be subjective.
    (In this case, how do we OPTIMIZE HYPER-PARAMETERS?)

You need to read more on K - Means clustering


How do we choose the best cluster number?
    We need to plot the clusters in a scatter plot and see which ones OVERLAP,
    keeping in mind that WE WANT OUR CLUSTERS TO BE AS DISTINCT AS POSSIBLE.
    
    Remember that even if we wanted to plot all 11 clusters, we would need
    a scatter plot with 11 dimentions. This is difficult to visualize.
    
    We need to use a dimentionality reduction technique in order to be able to 
    reduce the number of clusters.
    
    We would use: Canonical Discriminate Analysis (CDA).
    You must read on CDA!
    
    In CDA, we reduce dimentionality by creating NEW, fewer VARIABLES, which are
    a LINEAR combination of the original variables.
    
    In CDA, we obtain a first CANONICAL VARIABLE, which explains the largest proportion 
    of the TOTAL variance. 
    Next, we obtain the second canonical variable, which explains the next largest
    proportion of the TOTAL variance, etc.
    
    Usually, the first two NEW variables explain the majority of the variation.
    
    In python, we have Principal Component Analysis (PCA), which can help us to perform
    our CDA.


In the analysis below, we would be deploying our model in two cases:
    1. When K=2 (i.e., 2 clusters solution)
    2. When K=3 (i.e., 3 clusters solution)
    
    We would not deploy our model for K=6, even thought our elbow curve
    bent ak K=6. This id because 6 clusters is quite much   (?)
    
    
"""


"""
7. DEVELOPING SOLUTIONS FOR K=2 AND K=3 clusters
----------------------------------------------------------
"""
# Let us create a function to find the n cluster solution.
# this way, we can use it to visualize any number of clusters


def visualize_n_cluster_solution(n):
    # specify that you are working with 2 canonical variables
    pca_2 = PCA(2)

    # define and fit the model
    k_means_model = KMeans(n_clusters=n)
    k_means_model.fit(cluster_train)

    # fit.transform() method on training dataset
    plot_columns_pca = pca_2.fit_transform(cluster_train)

    # create the visualization
    plt.scatter(x=plot_columns_pca[:, 0],
                y=plot_columns_pca[:, 1],
                c=k_means_model.labels_, )

    plt.xlabel('Canonical Variable 1')
    plt.ylabel('Canonical Variable 2')
    plt.title(f'Scatter Plot for {n} Clusters Using 2 Canonical Variables')
    plt.show()


# visualize_n_cluster_solution(n=2)


"""
We see that 2 out of the 3 clusters in the vsualization of K=3 solution
are indeed a NEEDLESS separation of one cluster in the K=2 solution

We see that the overlap in the 6 cluster solution is terrible.
"""


"""
Now that we have identified that the 2 clusters solution is the best,
    we can say that our students can best be divided into 2 groups. 
    
    Now, it is time to see what students in each group HAVE IN COMMON.
            Read on K-Means clustering:
                1. What are the evaluation metrics?
                2. How do we interpret K-Means clustering results?
    
"""
