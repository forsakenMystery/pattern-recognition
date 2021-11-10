import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

a = 6
m = np.array([[0, 0, 0], [a, 0, 0], [a/2, a/2, 0], [0, a, 0], [-a/2, a/2, 0], [-a, 0, 0], [-a/2, -a/2, 0], [0, -a, 0], [a/2, -a/2, 0]])
n = 100
s = np.eye(3)
s[0, 0] = .5
s[1, 1] = .5
s[2, 2] = .01

ss = np.eye(3)
ss[2, 2] = .01
sss = np.array([s, ss, ss, ss, ss, ss, ss, ss, ss])


def data_generator(m, s, n):
    x = []
    for i in range(m.shape[0]):
        generate = np.random.multivariate_normal(m[i, :].T, s[i, :], n)
        # plt.scatter(generate[:, 0], generate[:, 1])
        # plt.show()
        if len(x) is 0:
            x = generate
        else:
            x = np.vstack([x, generate])
    y = np.hstack([np.ones([1, n]), 2*np.ones([1, 8*n])])
    # y = np.hstack([np.ones([1, n]), 2*np.ones([1, 7*n]), 3*np.ones([1, n])])
    return x, y.T.ravel()


x, y = data_generator(m, sss, n)
print(x.shape)

c1 = np.where(y == 1)
c2 = np.where(y == 2)
c3 = np.where(y == 3)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[c1, 0], x[c1, 1], x[c1, 2], label="class 1")
ax.scatter(x[c2, 0], x[c2, 1], x[c2, 2], label="class 2")
# ax.scatter(x[c3, 0], x[c3, 1], x[c3, 2], label="class 3")
plt.title("data")
ax.legend()
plt.show()

# clf = LinearDiscriminantAnalysis(solver="eigen", n_components=2)
# lda_x = clf.fit_transform(x, y)

# print(clf.covariance_)
# print(clf.explained_variance_ratio_)
# print(clf.n_components)
# print(clf.coef_)
# print(x.shape)
# print(lda_x.shape)
# plt.scatter(lda_x[c1, 0], lda_x[c1, 1])
# plt.scatter(lda_x[c2, 0], lda_x[c2, 1])
# plt.scatter(lda_x[c3, 0], lda_x[c3, 1])
# plt.show()


Mu1 = np.mean(x[c1], axis=0, keepdims=True)
Mu2 = np.mean(x[c2], axis=0, keepdims=True)

Mu1 = Mu1.T
Mu2 = Mu2.T


S1 = np.cov(x[c1].T)
S2 = np.cov(x[c2].T)


Sw = S1 + S2
Sb = (Mu1-Mu2).dot((Mu1-Mu2).T)


print(Mu2.shape)
print(Sw.shape)
value, vector = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))

print(value)
print("vector: ", vector)
print(vector[:, 0])
pairs = [(np.abs(value[i]), vector[:, i]) for i in range(len(value))]
pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
print("==============")
for pair in pairs:
    print(pair[0])

eigen_value_sums = sum(value)
print('Explained Variance')
for i, pair in enumerate(pairs):
    print('Eigenvector {}: {}'.format(i, (pair[0]/eigen_value_sums).real))

w_matrix = np.hstack((pairs[0][1].reshape(3, 1), pairs[1][1].reshape(3, 1))).real
print(w_matrix)
X1_lda = np.array(x[c1].dot(w_matrix))
X2_lda = np.array(x[c2].dot(w_matrix))
print(X1_lda.shape)
plt.scatter(X1_lda[:, 0], X1_lda[:, 1])
plt.scatter(X2_lda[:, 0], X2_lda[:, 1])
plt.show()
