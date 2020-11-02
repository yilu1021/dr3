from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
import matplotlib.pyplot as plt
import numpy as np
class ClusterBase:
    def calc_performance_score(self, algo_type: str, predicted, y_train):
        homo_score = homogeneity_score(y_train, predicted)
        complete_socre = completeness_score(y_train, predicted)
        adjusted_mute_info_score = adjusted_mutual_info_score(y_train, predicted)
        print(algo_type + ' homo_score ' + "{:.2f}".format(homo_score))
        print(algo_type + ' complete_socre ' + "{:.2f}".format(complete_socre))
        print(algo_type + ' adjusted_mute_info_score ' + "{:.2f}".format(adjusted_mute_info_score))

    def plot_homo(self, predicted, target_k, alg_type: str, subplot_x, subplot_y, y_train, data_set_tye: str):
        fig, axes = plt.subplots(subplot_x, subplot_y)
        fig.tight_layout()
        fig.set_size_inches(15, 10)

        for cluster in list(range(target_k)):
            ax = axes[cluster // subplot_y][cluster % subplot_y]
            indices = np.where(predicted == cluster)
            true_labels = y_train[indices]
            uniqueValues, occurCount = np.unique(true_labels, return_counts=True)
            uniqueValuesSorted = uniqueValues[np.argsort(occurCount)[::-1]]
            occurCountSorted = occurCount[np.argsort(occurCount)[::-1]]
            ax.set_title("Cluster " + str(cluster), fontdict={'fontsize': 10, 'fontweight': 'bold'})
            ax.pie(occurCountSorted, labels=uniqueValuesSorted,
                    shadow=True, startangle=90)

        plt.savefig('plots/' + data_set_tye + '_' + alg_type + '_homo.png')
        plt.close()

    def plot_completeness(self, predicted, alg_type: str, y_train, data_set_tye: str, number_of_target_class: int):
        fig, axes = plt.subplots(2, 5)
        fig.tight_layout()
        fig.set_size_inches(15, 10)
        for number in list(range(number_of_target_class)):
            ax = axes[number // 5][number % 5]
            indices = np.where(y_train == number)
            predicted_labels = predicted[indices]
            uniqueValues, occurCount = np.unique(predicted_labels, return_counts=True)
            uniqueValuesSorted = uniqueValues[np.argsort(occurCount)[::-1]]
            occurCountSorted = occurCount[np.argsort(occurCount)[::-1]]
            ax.set_title("Label: " + str(number), fontdict={'fontsize': 10, 'fontweight': 'bold'})
            ax.pie(occurCountSorted, labels=uniqueValuesSorted,
                    shadow=True, startangle=90)
        plt.savefig('plots/' + data_set_tye + '_' + alg_type + '_completeness.png')
        plt.close()