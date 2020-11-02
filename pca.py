import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
import numpy as np

class PCAAlgo():
    def plot_before_after_for_MNIST(self, y_train, X_train, data_set_type, reconstructed):
        _, axes = plt.subplots(1, 10)
        for digit in range(10):
            ax = axes[digit]
            index = np.where(y_train == digit)[0][0]
            x = X_train[index]
            ax.set_axis_off()
            ax.imshow(x.reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.savefig('plots/' + data_set_type + '_pca_sample_original_image_for_each_digit.png')
        plt.close()

        _, axes = plt.subplots(1, 10)
        for digit in range(10):
            ax = axes[digit]
            index = np.where(y_train == digit)[0][0]
            x = reconstructed[index]
            ax.set_axis_off()
            ax.imshow(x.reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.savefig('plots/' + data_set_type + '_pca_sample_reconstructed_image_for_each_digit.png')
        plt.close()

    def run_pca(self, number_of_features: int, target_component_n: int, X_train, y_train, data_set_type, isMNIST: bool):
        components_range = range(1, number_of_features+1)
        total_explained_variance_ratio = []
        eigenvalues = []
        reconstruction_losses = []

        for n_components in components_range:
            pca = PCA(n_components=n_components, random_state=42)
            x_train_transformed = pca.fit_transform(X_train)
            explained_variance_sum = pca.explained_variance_ratio_.sum()
            total_explained_variance_ratio.append(explained_variance_sum)
            eigenvalues.append(pca.explained_variance_)
            reconstructed = pca.inverse_transform(x_train_transformed)
            reconstruction_loss = ((X_train - reconstructed) ** 2).mean()
            reconstruction_losses.append(reconstruction_loss)
            if n_components == target_component_n:

                # plt.matshow(pca.components_, cmap='viridis')
                #
                # plt.yticks([0, 1], ["First component", "Second component"])
                # plt.colorbar()
                # # plt.xticks(range(len(cancer.feature_names)),
                # #            cancer.feature_names, rotation=60, ha='left')
                # plt.xlabel("Feature")
                # plt.ylabel("Principal components")
                # plt.show()
                # plt.close()

                colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
                          "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]
                xlim_min = x_train_transformed[:, 0].min()
                xlim_max = x_train_transformed[:, 0].max()
                ylim_min = x_train_transformed[:, 1].min()
                ylim_max = x_train_transformed[:, 1].max()

                for j in range(4):

                    plt.figure(figsize=(10, 10))

                    plt.xlim(xlim_min, xlim_max)
                    plt.ylim(ylim_min, ylim_max)

                    for i in range(len(X_train)):
                        # actually plot the digits as text instead of using scatter
                        plt.text(x_train_transformed[i, j], x_train_transformed[i, j+1], str(y_train[i]),
                                 color=colors[y_train[i]],
                                 fontdict={'weight': 'bold', 'size': 9})
                    plt.title('PC ' + str(j) + " and " + " PC " + str(j+1))
                    plt.xlabel('PC ' + str(j) )
                    plt.ylabel('PC ' + str(j + 1))
                    plt.savefig('plots/' + data_set_type + '_pca_' + str(j) + 'th_two_components_illustration.png')
                    plt.close()

                if isMNIST:
                    self.plot_before_after_for_MNIST(y_train=y_train, X_train=X_train, data_set_type=data_set_type,
                                                     reconstructed=reconstructed)


                if isMNIST:
                    fig, axes = plt.subplots(4, 7, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
                    for comp in range(target_component_n):
                        ax = axes[comp // 7][comp % 7]
                        ax.imshow(pca.components_[comp].reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
                        ax.set_title("{}. component".format(comp))
                    plt.savefig('plots/' + data_set_type + '_pca_' + str(n_components) + '_components_graph_' + '.png')
                    plt.close()


        plt.xlabel("Number of Components")
        title = 'MNIST'
        if data_set_type != 'mnist':
            title = 'Wine'
        plt.ylabel("Total Explained Variance Ratio")
        plt.title(title + ": Total Explained Variance Ratio for various Number of Component")
        xticks_names = []
        for n in list(components_range):
            xticks_names.append(str(n))
        plt.plot(components_range, total_explained_variance_ratio, marker='o')
        plt.savefig('plots/' + data_set_type + '_pca_explained_ratrio_over_component.png')
        plt.close()

        plt.xlabel("Number of Components")
        plt.ylabel("Reconstruction Error")
        plt.title(title + ": Reconstruction Error for various Number of Component")
        xticks_names = []
        for n in list(components_range):
            xticks_names.append(str(n))
        plt.plot(components_range, reconstruction_losses, marker='o')
        # plt.axvline(x=9, color="red", linestyle="--")
        plt.savefig('plots/' + data_set_type + '_pca_reconstruction_error_over_component.png')
        plt.close()


        target_comp_index = components_range.index(target_component_n)
        target_eigenvalues = eigenvalues[target_comp_index]
        x_pos = []
        for i in range(target_component_n):
            x_pos.append(i + 1)
        plt.bar(x_pos, target_eigenvalues, color='green')
        plt.xlabel("Component # (e.g., 1st component, 2nd component etc)")
        plt.ylabel("Eigen Values")
        plt.title(title + ": Eigenvalues For Each Component")
        plt.savefig('plots/' + data_set_type + '_pca_eigenvalues.png')
        plt.close()

        return reconstruction_losses