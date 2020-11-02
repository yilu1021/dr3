from typing import List
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import itertools
import numpy as np
from clusterbase import ClusterBase
class GaussianMix(ClusterBase):
    def plot(self, type: str, bics, k_range, cov_types, data_set_type: str):
        bics = np.array(bics)
        bars = []
        color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                      'darkorange'])
        plt.figure(figsize=(8, 6))
        for i, (cv_type, color) in enumerate(zip(cov_types, color_iter)):
            xpos = np.array(k_range) + .2 * (i - 2)
            bars.append(plt.bar(xpos, bics[i * len(k_range):
                                          (i + 1) * len(k_range)],
                                width=.2, color=color))
        plt.xticks(k_range)
        plt.ylim([bics.min() * 1.01 - .01 * bics.max(), bics.max()])
        title = 'MNIST'
        if data_set_type != 'mnist':
            title = 'Wine'
        plt.title(title + " " + type + ' score per model')
        xpos = np.mod(bics.argmin(), len(k_range)) + .65 + \
               .2 * np.floor(bics.argmin() / len(k_range))

        # plt.text(xpos, bics.min() * 0.97 + .03 * bics.max(), '*', fontsize=14)
        plt.xlabel('Number of K')
        plt.legend([b[0] for b in bars], cov_types)
        plt.savefig('plots/' + data_set_type + '_gaussian_' + type + '.png')
        plt.close()


    def run(self, k_range, X, data_set_type: str, target_k: int, target_cov: str, y_train, number_of_target_class):
        cov_types = ['spherical', 'tied', 'diag', 'full']
        results = []
        bics = []
        aics = []
        for cov_type in cov_types:
            for k in k_range:
                estimator = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=0)
                predict_labels = estimator.fit_predict(X)
                bic = estimator.bic(X)
                aic = estimator.aic(X)

                # bic = -2 * estimator.score(X) * X.shape[0]
                # aic = -2 * estimator.score(X) * X.shape[0]

                bics.append(bic)
                aics.append(aic)
                results.append({
                    'k': k,
                    'cov_type': cov_type,
                    'estimator': estimator,
                    'bic': bic,
                    'aic': aic,
                    'predict_labels': predict_labels
                })
        bics = np.array(bics)
        self.plot('BIC', bics, k_range, cov_types, data_set_type)
        aics = np.array(aics)
        self.plot('AIC', aics, k_range, cov_types, data_set_type)

        results = list(filter(lambda x: x['k'] == target_k and x['cov_type'] == target_cov, results))[0]
        predicted = results['predict_labels']

        algo_type = 'gaussian'

        self.plot_homo(predicted=predicted, target_k=target_k, alg_type=algo_type, subplot_x=3, subplot_y=4,
                       y_train=y_train, data_set_tye=data_set_type)

        self.plot_completeness(predicted=predicted, alg_type=algo_type, y_train=y_train, number_of_target_class=number_of_target_class, data_set_tye=data_set_type)

        self.calc_performance_score(algo_type=algo_type, predicted=predicted, y_train=y_train)
