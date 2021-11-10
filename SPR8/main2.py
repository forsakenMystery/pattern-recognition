import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics


n = [600, 200, 200, 100]
noise = .5
X = []
Y = []
R = 6
mini = -R
maxi = R
step = (maxi-mini)/((n[0]//2)-1)
for i in np.arange(mini, maxi, step):
    X.append([i, np.sqrt(R**2-i**2)+noise*(np.random.rand()-.5)])
    X.append([i, -np.sqrt(R**2-i**2)+noise*(np.random.rand()-.5)])
    Y.append(1)
    Y.append(1)
a = 3
b = 1
mini = -a
maxi = a
step = (maxi-mini)/((n[1]//2)-1)
for i in np.arange(mini, maxi, step):
    X.append([i, b*np.sqrt(1-i**2/a**2)+noise*(np.random.rand()-.5)])
    X.append([i, -b*np.sqrt(1-i**2/a**2)+noise*(np.random.rand()-.5)])
    Y.append(2)
    Y.append(2)
mini = -7
maxi = 7
step = (maxi-mini)/(n[2]-1)
coord = 8
for i in np.arange(mini, maxi, step):
    X.append([coord+noise*(np.random.rand()-.5), i+noise*(np.random.rand()-.5)])
    Y.append(3)
R = 3
center = 13
mini = center - R
maxi = center + R
step = (maxi - mini)/(n[3]-1)
for i in np.arange(mini, maxi, step):
    X.append([i, -np.sqrt(R**2-(i - center)**2)+noise*(np.random.rand()-.5)])
    Y.append(4)

X = np.array(X)
Y = np.array(Y)
for i in np.unique(Y):
    c = np.where(Y == i)
    plt.scatter(X[c, 0], X[c, 1], label="class "+str(i))
plt.legend()
plt.show()


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


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
    plt.scatter(estimator.cluster_centers_[:, 0], estimator.cluster_centers_[:, 1], marker='X', color='black')
    plt.title("algorithm is " + name + " " + op)
    plt.legend()
    plt.show()
    print("=====================")


k = np.unique(Y).shape[0]
cluster = k
clf = KMeans(n_clusters=k, init="random", n_init=k)
benchmark(clf, "k means", (X, Y), "k=%d" % k)