from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
class LDAAlgo():
    def run_lda(self, ida_target_n, ida_component_range, X_train, y_train, X_test, y_test, data_set_name):
        total_explained_variance_ratio = []
        components_range = range(ida_component_range)
        for n_component in components_range:
            lda = LinearDiscriminantAnalysis(n_components=n_component)
            X_lda = lda.fit_transform(X_train, y_train)
            explained_variance_sum = lda.explained_variance_ratio_.sum()
            total_explained_variance_ratio.append(explained_variance_sum)

        plt.xlabel("Number of Components")
        plt.ylabel("Total Explained Variance Ratio")
        plt.title("Total Explained Variance Ratio for various Number of Component")
        xticks_names = []
        for n in list(components_range):
            xticks_names.append(str(n))
        plt.plot(components_range, total_explained_variance_ratio, marker='o')
        plt.savefig('plots/' + data_set_name + '_lda_explained_ratrio_over_component.png')
        plt.close()

        pca = PCA(n_components=ida_target_n, random_state=42)
        X_pca = pca.fit_transform(X_train, y_train)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.scatter(
            X_pca[:,0],
            X_pca[:,1],
            c=y_train,
            cmap='rainbow',
            alpha=0.7,
            edgecolors='b'
        )
        plt.savefig('plots/' + data_set_name + '_lda_pca_transformed_x_comparison.png')
        plt.close()

        target_component = ida_target_n
        lda = LinearDiscriminantAnalysis(n_components=target_component)
        X_lda = lda.fit_transform(X_train, y_train)

        plt.xlabel('LD1')
        plt.ylabel('LD2')
        plt.scatter(
            X_lda[:,0],
            X_lda[:,1],
            c=y_train,
            cmap='rainbow',
            alpha=0.7,
            edgecolors='b'
        )
        plt.savefig('plots/' + data_set_name + '_lda_transformed_x.png')
        plt.close()

        y_pred = lda.predict(X_test)
        acc_score = accuracy_score(y_test, y_pred)
        print("LDA acc_score " + str(acc_score))