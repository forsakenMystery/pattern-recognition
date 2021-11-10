from sklearn import svm
import numpy as np
X = [[2, 0], [5, 1], [4, 2], [5, -1], [8, 0], [10, 4]]
y = [1, 1, 1, -1, -1, -1]
clf = svm.SVC()
clf.fit(X, y)
import matplotlib.pyplot as plt
X = np.array(X)
print(X[:, 0])
plt.scatter(X[:, 0], X[:, 1])
plt.show()