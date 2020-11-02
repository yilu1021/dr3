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
from nn import DR_NN
from sklearn.cluster import KMeans

class DatasetBase():
    def __init__(self, X_train, X_test, y_train, y_test, k_means_range, k_means_target, pca_target, em_k_range,
                 em_target, data_set_name, ica_target_n, ica_target_n_compare, rp_total_tries, ida_component_range,
                 ida_target_n, isMNIST, em_target_cov):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.k_means_range = k_means_range
        self.k_means_target = k_means_target
        self.pca_target = pca_target
        self.em_k_range = em_k_range
        self.em_target_cov = em_target_cov
        self.em_target = em_target
        self.data_set_name = data_set_name
        self.ica_target_n = ica_target_n
        self.ica_target_n_compare = ica_target_n_compare
        self.rp_total_tries = rp_total_tries
        self.ida_component_range = ida_component_range
        self.ida_target_n = ida_target_n
        self.isMNIST = isMNIST

        self.number_of_target_class = len(np.unique(self.y_train))
        self.number_of_features = self.X_train.shape[1]

        # self.X_train = StandardScaler().fit_transform(self.X_train)
        # self.X_test = StandardScaler().fit_transform(self.X_test)

    # def run_k_means_with_preset_centroid(self):
    #     init = []
    #     for digit in list(range(0, 10)):
    #         sample_index = np.where(self.y_train == digit)[0][0]
    #         init.append(self.X_train[sample_index])
    #     cluster_labels, silhouette_avg, cluster_centers = KMeansAlgo().run_silhouette_analysis_preset_centroid(
    #         X=self.X_train, init=np.array(init))
    #
    #     homo_score = homogeneity_score(self.y_train, cluster_labels)
    #     complete_score = completeness_score(self.y_train, cluster_labels)
    #     adjusted_mute_info_score = adjusted_mutual_info_score(self.y_train, cluster_labels)
    #     print(homo_score)

    def run_k_means(self, X=None, data_set_type=None, turned_target_k=None):
        if X is None:
            X = self.X_train
        if data_set_type is None:
            data_set_type = self.data_set_name

        target_k = self.k_means_target
        if turned_target_k is not None:
            target_k = turned_target_k
        KMeansAlgo().run_k_means(X_train=X, target_k=target_k, data_set_type=data_set_type, y_train=self.y_train,
                                 k_range=self.k_means_range,
                                 number_of_target_class=self.number_of_target_class)

    def run_em(self, X=None, data_set_type=None, turned_target_k=None):
        if X is None:
            X = self.X_train

        if data_set_type is None:
            data_set_type = self.data_set_name
        target_k = self.em_target
        if turned_target_k is not None:
            target_k = turned_target_k
        GaussianMix().run(k_range=self.em_k_range, X=X, data_set_type=data_set_type,
                                    target_k=target_k,
                                    target_cov=self.em_target_cov, y_train=self.y_train,
                          number_of_target_class=self.number_of_target_class)

    def run_pca(self):
        PCAAlgo().run_pca(number_of_features=self.number_of_features, target_component_n=self.pca_target,
                          X_train=self.X_train, y_train=self.y_train, data_set_type=self.data_set_name, isMNIST=self.isMNIST)
        pca = PCA(n_components=self.pca_target, random_state=42)
        x_train_transformed = pca.fit_transform(self.X_train)
        self.run_k_means(X=x_train_transformed, data_set_type=self.data_set_name + '_after_pca')
        turned_target_k = None
        if self.data_set_name == 'mnist':
            turned_target_k = 8
        self.run_em(X=x_train_transformed, data_set_type=self.data_set_name + '_after_pca', turned_target_k=turned_target_k)


    def run_ica(self):
        ICA().run_ica(number_of_features=self.number_of_features, target_n=self.ica_target_n,
          target_n_compare=self.ica_target_n_compare,
                     X_train=self.X_train, isMNIST=self.isMNIST, data_set_type=self.data_set_name)
        turned_target_k = None
        if self.data_set_name == 'mnist':
            turned_target_k = 9
        transformer = FastICA(n_components=self.ica_target_n, random_state=42, whiten=True, tol=0.1, max_iter=500)
        X_transformed = transformer.fit_transform(self.X_train)

        self.run_k_means(X=X_transformed, data_set_type=self.data_set_name + '_after_ica')
        self.run_em(X=X_transformed, data_set_type=self.data_set_name + '_after_ica', turned_target_k=turned_target_k)

    def run_rp(self):
        # rp_total_tries, number_of_features, X_train, pca_reconstruction_losses, data_set_name
        pca_reconstruction_losses = PCAAlgo().run_pca(number_of_features=self.number_of_features,
                                                      target_component_n=self.pca_target,
                          X_train=self.X_train, y_train=self.y_train, data_set_type=self.data_set_name, isMNIST=self.isMNIST)
        RP().run_rp(rp_total_tries=self.rp_total_tries, number_of_features=self.number_of_features,
                    X_train=self.X_train, pca_reconstruction_losses=pca_reconstruction_losses, data_set_name=self.data_set_name)

        transformer = InvertibleRandomProjection(n_components=self.pca_target)
        X_transformed = transformer.fit_transform(self.X_train)
        turned_target_k = None
        if self.data_set_name == 'mnist':
            turned_target_k = 10
        self.run_k_means(X=X_transformed, data_set_type=self.data_set_name + '_after_rp')

        turned_target_k = None
        if self.data_set_name == 'mnist':
            turned_target_k = 7
        self.run_em(X=X_transformed, data_set_type=self.data_set_name + '_after_rp')

    def run_lda(self):
        LDAAlgo().run_lda(ida_target_n=self.ida_target_n, ida_component_range=self.ida_component_range,
                          X_train=self.X_train, y_train=self.y_train, X_test=self.X_test, y_test=self.y_test,
                          data_set_name=self.data_set_name)

        transformer = LinearDiscriminantAnalysis(n_components=self.ida_target_n)
        X_transformed = transformer.fit_transform(self.X_train, self.y_train)

        self.run_k_means(X=X_transformed, data_set_type=self.data_set_name + '_after_lda')
        self.run_em(X=X_transformed, data_set_type=self.data_set_name + '_after_lda')

    def run_dr_nn(self):
        # DR_NN().run_nn(X_train_full=self.X_train, X_test=self.X_test, y_train_full=self.y_train, y_test=self.y_test,
        #                type="MNIST_DIGIT_ORIGINAL", optimal_hidden_size=40)
        #
        # pca = PCA(n_components=self.pca_target, random_state=42)
        # pca_x_train_transformed = pca.fit_transform(self.X_train)
        # DR_NN().run_nn(X_train_full=pca_x_train_transformed, X_test=pca.transform(self.X_test), y_train_full=self.y_train,
        #                y_test=self.y_test,
        #                type="MNIST_DIGIT_PCA_Transformed", optimal_hidden_size=20)
        #
        transformer = FastICA(n_components=self.ica_target_n, random_state=42, whiten=True, tol=0.1, max_iter=500)
        ica_X_transformed = transformer.fit_transform(self.X_train)
        DR_NN().run_nn(X_train_full=ica_X_transformed, X_test=transformer.transform(self.X_test), y_train_full=self.y_train,
                       y_test=self.y_test,
                       type="MNIST_DIGIT_ICA_Transformed", optimal_hidden_size=20)

        transformer = LinearDiscriminantAnalysis(n_components=self.ida_target_n)
        lda_X_transformed = transformer.fit_transform(self.X_train, self.y_train)
        DR_NN().run_nn(X_train_full=lda_X_transformed, X_test=transformer.transform(self.X_test), y_train_full=self.y_train,
                       y_test=self.y_test,
                       type="MNIST_DIGIT_lda_Transformed", optimal_hidden_size=20)

        transformer = InvertibleRandomProjection(n_components=self.pca_target)
        rp_X_transformed = transformer.fit_transform(self.X_train)
        DR_NN().run_nn(X_train_full=rp_X_transformed, X_test=transformer.transform(self.X_test), y_train_full=self.y_train,
                       y_test=self.y_test,
                       type="MNIST_DIGIT_rp_Transformed", optimal_hidden_size=20)

    def run_clust_nn(self):

        # DR_NN().run_nn(X_train_full=self.X_train, X_test=self.X_test, y_train_full=self.y_train, y_test=self.y_test,
        #                type="MNIST_DIGIT_ORIGINAL", optimal_hidden_size=8)

        clusterer = KMeans(n_clusters=self.k_means_target, random_state=42)
        clusterer.fit(self.X_train)
        X_train_predict = clusterer.predict(self.X_train)
        X_test_predict = clusterer.predict(self.X_test)

        X_train_new_feat = []
        X_test_new_feat = []
        for n in range(X_train_predict.shape[0]):
            x_copy = np.copy(self.X_train[n])
            new_x = np.concatenate([x_copy, [X_train_predict[n]]])
            X_train_new_feat.append(new_x)

        for n in range(X_test_predict.shape[0]):
            x_copy = np.copy(self.X_test[n])
            new_x = np.concatenate([x_copy, [X_test_predict[n]]])
            X_test_new_feat.append(new_x)

        X_train_new_feat = np.array(X_train_new_feat)
        X_test_new_feat = np.array(X_test_new_feat)

        DR_NN().run_nn(X_train_full=X_train_new_feat, X_test=X_test_new_feat, y_train_full=self.y_train, y_test=self.y_test,
                       type="MNIST_DIGIT_K_Mean_New_Feat", optimal_hidden_size=20)

        clusterer = GaussianMixture(n_components=self.em_target, random_state=0, covariance_type=self.em_target_cov)
        clusterer.fit(self.X_train)
        X_train_predict = clusterer.predict(self.X_train)
        X_test_predict = clusterer.predict(self.X_test)

        X_train_new_feat = []
        X_test_new_feat = []
        for n in range(X_train_predict.shape[0]):
            x_copy = np.copy(self.X_train[n])
            new_x = np.concatenate([x_copy, [X_train_predict[n]]])
            X_train_new_feat.append(new_x)

        for n in range(X_test_predict.shape[0]):
            x_copy = np.copy(self.X_test[n])
            new_x = np.concatenate([x_copy, [X_test_predict[n]]])
            X_test_new_feat.append(new_x)

        X_train_new_feat = np.array(X_train_new_feat)
        X_test_new_feat = np.array(X_test_new_feat)
        DR_NN().run_nn(X_train_full=X_train_new_feat, X_test=X_test_new_feat, y_train_full=self.y_train, y_test=self.y_test,
                       type="MNIST_DIGIT_EM_New_Feat", optimal_hidden_size=20)

    def run(self):
        print("***DATASET " + self.data_set_name + "***")
        # print('k-means')
        # self.run_k_means()
        # print('EM')
        # self.run_em()
        # print('PCA')
        # self.run_pca()
        # print('ICA')
        # self.run_ica()
        # print('RP')
        # self.run_rp()
        # print('LDA')
        # self.run_lda()

        # print('DR NN')
        # self.run_dr_nn()

        print('Cluster NN')
        self.run_clust_nn()