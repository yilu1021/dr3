from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from kmeans import KMeansAlgo
from gaussianmix import GaussianMix
import numpy as np
import matplotlib.pyplot as plt
from invertiblerp import InvertibleRandomProjection
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn import random_projection

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy
from sklearn.metrics import accuracy_score
from pca import PCAAlgo
from ica import ICA
from rp import RP
from lda import LDAAlgo
from dataset_base import DatasetBase

def get_MNIST():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    targets = digits.target

    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2,
                                                        stratify=targets, random_state=42)
    k_means_range = range(2, 14)
    k_means_target = 9
    pca_target = 28
    em_k_range = range(2, 20)
    em_target = 12
    data_set_name = 'mnist'
    ica_target_n = 54
    ica_target_n_compare = 20
    rp_total_tries = 25
    ida_component_range = 10
    ida_target_n = 7
    isMNIST = True
    return DatasetBase(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, k_means_range=k_means_range,
                       k_means_target=k_means_target,
                       pca_target=pca_target,
                       em_k_range=em_k_range,
                       em_target=em_target, data_set_name=data_set_name, ica_target_n=ica_target_n, ica_target_n_compare=ica_target_n_compare,
                       rp_total_tries=rp_total_tries,
                       ida_component_range=ida_component_range,
                       ida_target_n=ida_target_n,
                       isMNIST=isMNIST, em_target_cov='diag')

def get_Wine():
    data, targets = datasets.load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.3,
                                                        stratify=targets, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    k_means_range = range(2, 10)
    k_means_target = 3
    pca_target = 10
    em_k_range = range(2, 10)
    em_target = 3
    data_set_name = 'wine'
    ica_target_n = 10
    ica_target_n_compare = 5
    rp_total_tries = 25
    ida_component_range = 3
    ida_target_n = 2
    isMNIST = False
    return DatasetBase(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, k_means_range=k_means_range,
                       k_means_target=k_means_target,
                       pca_target=pca_target,
                       em_k_range=em_k_range,
                       em_target=em_target, data_set_name=data_set_name, ica_target_n=ica_target_n, ica_target_n_compare=ica_target_n_compare,
                       rp_total_tries=rp_total_tries,
                       ida_component_range=ida_component_range,
                       ida_target_n=ida_target_n,
                       isMNIST=isMNIST, em_target_cov='diag')

def main():
    mnist = get_MNIST()
    mnist.run()

    wine = get_Wine()
    wine.run()
    # MNIST().run_k_means()
    # MNIST().run_em()
    # MNIST().run_pca()
    # MNIST().run_ica()
    # MNIST().run_rp()
    # MNIST().run_lda()

    # MNIST().run_k_means_with_preset_centroid()

if __name__ == '__main__':
    main()




# Helpful Methods
# KMeans(n_clusters=k)
# GaussianMixture(n_components=k)
# GaussianRandomProjection
# Silhouette Analysis to determine a k for kmeans:
#     https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
# Calculate Intertia
# Elbow method
# https://www.scikit-yb.org/en/latest/api/cluster/elbow.html
# Silouette Scores
# K means clustering with Iris
# https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html

# Evaluating Clustering
# https://piazza.com/class/kdx36x23bcer4?cid=638

# To determine how to get number of components:
# - ICA: Kurtosis at a minimum. Good ICA would be higher Kurtosis and less Gaussian?
# - Random Projection: Reconstruction Error
# - PCA: Eigenvalue?

# EM implementation, use GaussianMixture
# part 5 - use the output of part 1.

# Office hour
# Since this is UL .. you cannot involve the labels,, during the learning process..
# However, when you are evaluating the performance, you can use your ground labels ...
# But you CAN use mwtrics: sum of swaured metrics, silhouette metric, BIC, etc... that do not really on ground truth ...
# There are metrics: adjusted mutually information, homogeniety, completeness, V score ... that give you some intuition using the ground label ..
# You have to have BOTH otherwise you LOSE points ...
#
# 1) Choose optimal number of points (no labels: use metrics like sum of squared errors, silhouette metric, BIC, etc)
#
# 2) Evaluate performance (yes labels: use metrics like adjusted mutual information, homogeneity completeness, V score, etc)

# Perspective of "runtime" and "performance" is helpful

#ICA might be helpful
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.fastica.html?highlight=ica#sklearn.decomposition.fastica

#Pair plot?
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html#sphx-glr-auto-examples-cluster-plot-kmeans-assumptions-py


# Helpful comments:
# https://gyazo.com/3b47b088ebf21792cddb592a2eef60d8

# what is the meaning of reconstruction?
# https://piazza.com/class/kdx36x23bcer4?cid=642


#AIC/BIC?
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html
# https://piazza.com/class/kdx36x23bcer4?cid=667

