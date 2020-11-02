from sklearn.model_selection import train_test_split
from typing import List
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from clusterbase import ClusterBase

class KMeansAlgo(ClusterBase):

    def run_silhouette_analysis_preset_centroid(self, X, init: ndarray):
        clusterer = KMeans(n_clusters=init.shape[0], init=init)
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        cluster_centers = clusterer.cluster_centers_
        return cluster_labels, silhouette_avg, cluster_centers

    def run_silhouette_analysis(self, k_range: List[int], X, random_state:int):
        k_avg_scores: List[float] = []
        k_sample_scores = []
        k_cluster_labels = []
        k_cluster_centers = []
        k_cluster_inertias = []
        for n_clusters in k_range:
            clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
            cluster_labels = clusterer.fit_predict(X)
            silhouette_avg = silhouette_score(X, cluster_labels)
            k_avg_scores.append(silhouette_avg)
            k_cluster_centers.append(clusterer.cluster_centers_)
            k_cluster_labels.append(cluster_labels)
            k_cluster_inertias.append(clusterer.inertia_)
            k_sample_scores.append(silhouette_samples(X, cluster_labels))
        return k_avg_scores, k_sample_scores, k_cluster_labels, k_cluster_centers, k_cluster_inertias

    def plot_silhouette(self, k_range, k_avg_scores, k_sample_scores, k_cluster_labels, k_cluster_centers,
                  k_cluster_inertias, X_train, data_set_type):
        min_score = np.amin(np.array(k_sample_scores))
        y_seperator_len = 10
        fig, axes = plt.subplots(3, 4)
        fig.tight_layout()
        fig.set_size_inches(20, 15)
        for i, k in enumerate(k_range):
            ax = axes[i // 4][i % 4]
            ax.set_xlim([min_score, 1])
            ax.set_ylim([0, len(X_train) + (k + 1) * y_seperator_len])
            y_lower = y_seperator_len
            for j in range(k):
                jth_cluster_silhouette_values = k_sample_scores[i][k_cluster_labels[i] == j]
                size_cluster_j = jth_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_j
                color = cm.nipy_spectral(float(j) / k)
                jth_cluster_silhouette_values_copy = np.copy(jth_cluster_silhouette_values)
                jth_cluster_silhouette_values_copy.sort()
                ax.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, jth_cluster_silhouette_values_copy,
                                  facecolor=color, edgecolor=color, alpha=0.7)
                ax.text(-0.05, y_lower + 0.5 * size_cluster_j, str(j))
                y_lower = y_upper + y_seperator_len
            ax.set_title("K = " + str(k) + "; " + "Avg Silhouette Value: " + "{:.2f}".format(
                k_avg_scores[i]), fontdict={'fontsize': 10, 'fontweight': 'bold'})
            ax.set_xlabel("Silhouette value")
            ax.set_ylabel("Cluster label")
            ax.axvline(x=k_avg_scores[i], color="red", linestyle="--")
        plt.savefig('plots/' + data_set_type + '_k_means_silhouette.png')
        plt.close()

    def plot_elbow(self, k_range, k_avg_scores, k_sample_scores, k_cluster_labels, k_cluster_centers,
                  k_cluster_inertias, data_set_type, target_k):
        plt.xlabel("K")
        plt.ylabel("Inertia")
        title = 'MNIST'
        if data_set_type != 'mnist':
            title = 'Wine'
        plt.title(title + " Inertia for various K")

        xticks_names = []
        for k in list(k_range):
            xticks_names.append(str(k))
        plt.xticks(k_range, xticks_names)
        plt.plot(k_range, k_cluster_inertias, marker='o')
        plt.axvline(x=target_k, color="red", linestyle="--")
        plt.savefig('plots/' + data_set_type + '_k_means_inertia.png')
        plt.close()

    def run_k_means(self, X_train, target_k: int, data_set_type: str, y_train, k_range, number_of_target_class):

        k_avg_scores, k_sample_scores, k_cluster_labels, k_cluster_centers, k_cluster_inertias = \
            self.run_silhouette_analysis(k_range=list(k_range), X=X_train, random_state=42)

        self.plot_silhouette(k_range=k_range, k_avg_scores=k_avg_scores,
                       k_sample_scores=k_sample_scores, k_cluster_labels=k_cluster_labels,
                       k_cluster_centers=k_cluster_centers, k_cluster_inertias=k_cluster_inertias, X_train=X_train,
                             data_set_type = data_set_type)

        self.plot_elbow(k_range=k_range, k_avg_scores=k_avg_scores,
                       k_sample_scores=k_sample_scores, k_cluster_labels=k_cluster_labels,
                       k_cluster_centers=k_cluster_centers, k_cluster_inertias=k_cluster_inertias, data_set_type =
                        data_set_type, target_k=target_k)

        predicted = k_cluster_labels[k_range.index(target_k)]

        algo_type = 'k_means'

        self.calc_performance_score(algo_type=algo_type, predicted=predicted, y_train=y_train)

        self.plot_homo(predicted=predicted, target_k=target_k, alg_type=algo_type, subplot_x=3, subplot_y=3,
                       y_train=y_train, data_set_tye=data_set_type)

        self.plot_completeness(predicted=predicted, alg_type=algo_type, y_train=y_train, data_set_tye=data_set_type,
                               number_of_target_class=number_of_target_class)
