from invertiblerp import InvertibleRandomProjection
import numpy as np
import matplotlib.pyplot as plt
class RP():
    def run_rp(self, rp_total_tries, number_of_features, X_train, pca_reconstruction_losses, data_set_name):
        total_tries = rp_total_tries
        components_range = range(1, number_of_features + 1)
        reconstruction_losses = []
        standard_devs = []
        for n_components in components_range:
            sub_reconstruction_losses = []
            for n_try in range(total_tries):
                transformer = InvertibleRandomProjection(n_components=n_components)
                X_transformed = transformer.fit_transform(X_train)
                reconstructed = transformer.inverse_transform(X_transformed)
                reconstruction_loss = ((X_train - reconstructed) ** 2).mean()
                sub_reconstruction_losses.append(reconstruction_loss)
            reconstruction_losses.append(np.array(sub_reconstruction_losses).mean())
            standard_devs.append(np.std(np.array(sub_reconstruction_losses)))


        print('RP standard devs')
        print(standard_devs)
        plt.xlabel("Number of Components")
        plt.ylabel("Reconstruction Error")
        title = "MNIST"
        if data_set_name != 'mnist':
            title = 'Wine'
        plt.title(title + ": Randmized Projection vs. PCA in Reconstruction Error")
        xticks_names = []
        for n in list(components_range):
            xticks_names.append(str(n))
        # plt.xticks(components_range, xticks_names)
        # plt.yticks()

        plt.plot(components_range, reconstruction_losses, marker='o', label='Randomized Projection')
        plt.plot(components_range, pca_reconstruction_losses, marker='x', label='PCA')
        plt.legend()

        # plt.axvline(x=9, color="red", linestyle="--")
        plt.savefig('plots/' + data_set_name + '_rp_reconstruction_error_over_component.png')
        plt.close()