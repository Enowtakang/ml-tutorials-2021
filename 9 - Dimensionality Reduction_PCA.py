"""
Dimensionality Reduction - PCA

    - Here, we derive a set of NEW FEATURES which are fewer than the
        Original Features while PRESERVING the VARIANCE in the data.
    - PCA is a common dimensionality reduction technique.
    - It is used in many applications like FACIAL RECOGNITION.
    - PCA is performed using LINEAR COMBINATIONS of the original
        features of the data.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA


"""
1. Load Data
"""
iris = load_iris()
X = iris.data
y = iris.target

"""
2. Instantiate and Use PCA
"""
pca_model = PCA(n_components=2, whiten=True)
pca_model.fit(X)

"""
please you need to read what the following results mean!!!
"""
# print(pca_model.components_)
# print(pca_model.explained_variance_ratio_)
# print(pca_model.explained_variance_ratio_.sum())

"""
3. Transform the features dataset
"""
x_pca = pca_model.transform(X)

"""
4. Plot iris data
"""


def plot_iris(data, target, target_names):

    colors = cycle('rgb')

    ids = range(len(target_names))

    plt.figure()

    for p, c, label in zip(ids, colors, target_names):

        plt.scatter(data[target == p, 0],
                    data[target == p, 1],
                    c=c,
                    label=label)

    plt.legend()
    plt.show()


# plot_iris(x_pca, iris.target, iris.target_names)

"""
Read more about PCA in Sklearn Documentation!!!!!!!!!!
"""
