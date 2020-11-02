from sklearn.decomposition import PCA, FastICA
import numpy as np
import scipy
import matplotlib.pyplot as plt

class ICA():

    def plot_components_for_MNIST_compare(self, transformer, target_n_compare, data_set_type, n_components):
        fig, axes = plt.subplots(4, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
        for comp in range(target_n_compare):
            ax = axes[comp // 5][comp % 5]
            ax.imshow(transformer.components_[comp].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
            ax.set_title("{}. component".format(comp))
        plt.savefig('plots/' + data_set_type + '_ica_' + str(n_components) + '_components_graph_' + '.png')
        plt.close()

    def plot_components_for_MNIST(self, transformer, n_components, target_n, data_set_type, target_n_compare):

        fig, axes = plt.subplots(7, 8, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
        for comp in range(target_n):
            ax = axes[comp // 8][comp % 8]
            ax.imshow(transformer.components_[comp].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
            ax.set_title("{}. component".format(comp))

        plt.savefig('plots/' + data_set_type + '_ica_' + str(n_components) + '_components_graph_' + '.png')
        plt.close()

    def plot_top_digits_activates_comp(self, X_transformed, target_n, X_train, data_set_type, n_components):
        for comp in range(target_n):
            target_comp = comp
            inds = np.argsort(X_transformed[:, target_comp])[::-1]
            fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                                     subplot_kw={'xticks': (), 'yticks': ()})
            fig.suptitle("Large component " + str(comp))
            for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
                ax.imshow(X_train[ind].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')

            plt.savefig('plots/' + data_set_type + '_ica_top_component_x_train/overall_component_' + str(
                n_components) + '_'
                        + 'specific_component_' + str(comp) + '.png')
            plt.close()

    def run_ica(self, number_of_features, target_n, target_n_compare, X_train, isMNIST, data_set_type):
        components_range = range(1, number_of_features)

        KURTOSIS_SCORES = []

        for n_components in components_range:
            transformer = FastICA(n_components=n_components, random_state=42, whiten=True, tol=0.1, max_iter=500)
            X_transformed = transformer.fit_transform(X_train)
            components = transformer.components_
            kurtosis = np.mean(scipy.stats.kurtosis(components, axis=0))
            KURTOSIS_SCORES.append(kurtosis)

            # if n_components == 10:
            #     reconstructed = transformer.inverse_transform(X_transformed)
            #     _, axes = plt.subplots(1, 10)
            #     for digit in range(10):
            #         ax = axes[digit]
            #         index = np.where(self.y_train == digit)[0][0]
            #         x = reconstructed[index]
            #         ax.set_axis_off()
            #         ax.imshow(x.reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
            #     plt.savefig('plots/' + 'mnist_ica_sample_reconstructed_image_for_each_digit.png')
            #     plt.close()
            if n_components == target_n_compare and isMNIST:
                self.plot_components_for_MNIST_compare(transformer=transformer, target_n_compare=target_n_compare,
                                                       data_set_type=data_set_type, n_components=n_components)

            if n_components == target_n:
                if isMNIST:
                    self.plot_components_for_MNIST(transformer=transformer,n_components=n_components, target_n=target_n,
                                               data_set_type=data_set_type, target_n_compare=target_n_compare)
                    self.plot_top_digits_activates_comp(X_transformed=X_transformed, target_n=target_n, X_train=X_train, data_set_type=data_set_type, n_components=n_components)

        plt.xlabel("Number of Componenets")
        plt.ylabel("Average Kurtosis")
        title = "MNIST"
        if data_set_type != "mnist":
            title = "Wine"
        plt.title(title + " Kurtosis for various number of Components")
        plt.plot(components_range, KURTOSIS_SCORES, marker='o')
        plt.axvline(x=target_n, color="red", linestyle="--")
        plt.savefig('plots/' + data_set_type + '_ica_kurtosis.png')
        plt.close()