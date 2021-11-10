import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics


class KMEANS_Scratch:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]
        for i in range(self.max_iter):
            self.classifications = {}
            for i in range(self.k):
                self.classifications[i] = []
            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
            prev_centroids = dict(self.centroids)
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False
            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


m = np.array([[0, 0], [5, 5]])
s1 = 1.5*np.eye(2)
s2 = np.eye(2)
s = np.array([s1, s2])


def benchmark(estimator, name, data, op=None):
    print("algorithm:", name)
    X, Y = data
    estimator.fit(X)
    print("homogeneity score: %.2f" % metrics.homogeneity_score(Y, estimator.labels_))
    print("completeness score: %.2f" % metrics.completeness_score(Y, estimator.labels_))
    print("jaccard score: %.2f" % metrics.jaccard_similarity_score(Y, estimator.labels_))
    print("normalized mutual information score: %.2f" % metrics.normalized_mutual_info_score(Y, estimator.labels_))

    pallet = get_cmap(np.unique(estimator.labels_).shape[0] * 5)
    print("silhouette score: %.2f" % metrics.silhouette_score(X, estimator.labels_))
    for i, c in enumerate(np.unique(estimator.labels_)):
        x = X[np.where(estimator.labels_ == c)]
        plt.scatter(x[:, 0], x[:, 1], c=pallet(i * 5), label=f"class number %d" % c)
    plt.title("algorithm is " + name + " " + op)
    plt.legend()
    plt.show()
    print("=====================")


def data_generator(m, s):
    x = np.random.multivariate_normal(m[0, :].T, s[0], 500)
    x = np.vstack([x, np.random.multivariate_normal(m[1, :].T, s[1], 15)])
    y = np.hstack([np.zeros([1, 500]), np.ones([1, 15])])
    return x, y


x, y = data_generator(m, s)
c1 = np.where(y == 0)
c2 = np.where(y == 1)
plt.scatter(x[c1, 0], x[c1, 1])
plt.scatter(x[c2, 0], x[c2, 1])
plt.show()

k = np.unique(y).shape[0]
cluster = k
k = 2
clf = KMeans(n_clusters=k, init="random", n_init=k)
benchmark(clf, "k means", (x, y.ravel()), "k=%d" % k)