import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

m = np.array([[-6, 6, 6], [6, 6, 6]])
# m = np.array([[-2, 2, 2], [2, 0, 0]])
n = 200
s = np.ones((3, 3))
s[0, 0] = .3
s[1, 1] = 9
s[2, 2] = 9


def data_generator(m, s, n):
    x = []
    for i in range(m.shape[0]):
        generate = np.random.multivariate_normal(m[i, :].T, s, n)
        # plt.scatter(generate[:, 0], generate[:, 1])
        # plt.show()
        if len(x) is 0:
            x = generate
        else:
            x = np.vstack([x, generate])
    y = np.hstack([np.ones([1, n]), -np.ones([1, n])])
    return x, y


x, y = data_generator(m, s, n)
pca = PCA(n_components=3)
pca.fit(x)
print(pca.explained_variance_ratio_)
x_prime = pca.transform(x)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
c1 = np.where(y == 1)
c2 = np.where(y == -1)
ax.scatter(x[c1, 0], x[c1, 1], x[c1, 2], label="class 1")
ax.scatter(x[c2, 0], x[c2, 1], x[c2, 2], label="class 2")
plt.title("data")
ax.legend()
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
plt.title("PCA")
ax1.scatter(x_prime[c1, 0], x_prime[c1, 1], x_prime[c1, 2], label="class 1")
ax1.scatter(x_prime[c2, 0], x_prime[c2, 1], x_prime[c2, 2], label="class 2")
ax1.legend()
plt.show()

plt.subplot(231)
plt.scatter(x[c1, 0], x[c1, 1], label="class 1")
plt.scatter(x[c2, 0], x[c2, 1], label="class 2")
plt.legend()
plt.subplots_adjust(wspace=.35, hspace=.25)
plt.title("X1 - X2")
plt.xlabel('X1')
plt.ylabel('X2')
plt.subplot(232)
plt.scatter(x[c1, 0], x[c1, 2], label="class 1")
plt.scatter(x[c2, 0], x[c2, 2], label="class 2")
plt.legend()
plt.title("X1 - X3")
plt.xlabel('X1')
plt.ylabel('X3')
plt.subplot(233)
plt.scatter(x[c1, 1], x[c1, 2], label="class 1")
plt.scatter(x[c2, 1], x[c2, 2], label="class 2")
plt.legend()
plt.title("X2 - X3")
plt.xlabel('X2')
plt.ylabel('X3')
plt.subplot(234)
plt.scatter(x_prime[c1, 0], x_prime[c1, 1], label="class 1")
plt.scatter(x_prime[c2, 0], x_prime[c2, 1], label="class 2")
plt.legend()
plt.title("Y1 - Y2")
plt.xlabel('Y1')
plt.ylabel('Y2')
plt.subplot(235)
plt.scatter(x_prime[c1, 0], x_prime[c1, 2], label="class 1")
plt.scatter(x_prime[c2, 0], x_prime[c2, 2], label="class 2")
plt.legend()
plt.title("Y1 - Y3")
plt.xlabel('Y1')
plt.ylabel('Y3')
plt.subplot(236)
plt.scatter(x_prime[c1, 1], x_prime[c1, 2], label="class 1")
plt.scatter(x_prime[c2, 1], x_prime[c2, 2], label="class 2")
plt.legend()
plt.title("Y2 - Y3")
plt.xlabel('Y2')
plt.ylabel('Y3')
plt.show()