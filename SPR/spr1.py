import numpy as np
import matplotlib.pyplot as plt

N = 1000
mu1 = [1, 1]
mu2 = [12, 8]
mu3 = [16, 1]
cov = 4*np.eye(2)

#
# mu1 = [1, 1]
# mu2 = [14, 7]
# mu3 = [16, 1]
# cov = np.array([[5, 3], [3, 4]])


mu1 = np.array(mu1).T
mu2 = np.array(mu2).T
mu3 = np.array(mu3).T


def generate_data(mean, covariance, num=1):
    return np.random.multivariate_normal(mean, covariance, num)


# L = np.linalg.cholesky(cov)
# uncorrelated = np.random.standard_normal((2, N))
# data2 = np.dot(L, uncorrelated) + np.array(mean).reshape(2, 1)
# both are valid ideas
# plt.scatter(data2[0, :], data2[1, :], c='green')


def sample_data(num=N):
    sampled = []
    y_sampled = []
    class1 = []
    class2 = []
    class3 = []
    for i in range(num):
        prob = np.random.randint(0, 99)
        if prob < 33:
            sampled.append(generate_data(mu1, cov)[0])
            y_sampled.append(0)
            class1.append(i)
        elif prob < 66:
            sampled.append(generate_data(mu2, cov)[0])
            y_sampled.append(1)
            class2.append(i)
        else:
            sampled.append(generate_data(mu3, cov)[0])
            y_sampled.append(2)
            class3.append(i)
    return np.array(sampled), np.array(y_sampled), class1, class2, class3


X, Y, c1, c2, c3 = sample_data()


def mahalanobis(x, data=None, mu=None, cov=None):
    if mu is None:
        if data is None:
            raise Exception('you should give me sth to work with')
        mu = np.mean(data)
    if cov is None:
        if data is None:
            raise Exception('you should give me sth to work with')
        cov = np.cov(data)
    centered = x - mu
    inv_cov = np.linalg.inv(cov)
    left_term = np.dot(centered, inv_cov)
    mahal = np.dot(left_term, centered.T)
    return mahal.diagonal()


mu_p1 = np.mean(X[c1], axis=0)
cov_p1 = np.cov(X[c1].T)
mc1 = mahalanobis(X, mu=mu_p1, cov=cov_p1)

mu_p2 = np.mean(X[c2], axis=0)
cov_p2 = np.cov(X[c2].T)
mc2 = mahalanobis(X, mu=mu_p2, cov=cov_p2)

mu_p3 = np.mean(X[c3], axis=0)
cov_p3 = np.cov(X[c3].T)
mc3 = mahalanobis(X, mu=mu_p3, cov=cov_p3)

classed_mahal = np.array([mc1, mc2, mc3])

fail = np.count_nonzero(Y - np.argmin(classed_mahal.T, axis=1))
print("Mahalanobis: ")
print(fail, "number of Misclassified which will be", fail/N, "% misclassified in total")
print((np.count_nonzero(Y[c1] - np.argmin(classed_mahal.T[c1], axis=1)))/len(c1), "% misclassified in class 1")
print((np.count_nonzero(Y[c2] - np.argmin(classed_mahal.T[c2], axis=1)))/len(c2), "% misclassified in class 2")
print((np.count_nonzero(Y[c3] - np.argmin(classed_mahal.T[c3], axis=1)))/len(c3), "% misclassified in class 3")
print("======================================================")
# origin point
origin = [0], [0]

print("Euclidean: ")
d1 = np.linalg.norm(X-mu_p1, axis=1)
d2 = np.linalg.norm(X-mu_p2, axis=1)
d3 = np.linalg.norm(X-mu_p3, axis=1)

classed_euc = np.array([d1, d2, d3])
fail = np.count_nonzero(Y - np.argmin(classed_euc.T, axis=1))
print(fail, "number of Misclassified which will be", fail/N, "% misclassified in total")
print((np.count_nonzero(Y[c1] - np.argmin(classed_euc.T[c1], axis=1)))/len(c1), "% misclassified in class 1")
print((np.count_nonzero(Y[c2] - np.argmin(classed_euc.T[c2], axis=1)))/len(c2), "% misclassified in class 2")
print((np.count_nonzero(Y[c3] - np.argmin(classed_euc.T[c3], axis=1)))/len(c3), "% misclassified in class 3")
print("======================================================")
print("Bayesian: ")

probability_c1 = len(c1)/N
print(probability_c1, "probability of first class")
probability_c2 = len(c2)/N
print(probability_c2, "probability of second class")
probability_c3 = len(c3)/N
print(probability_c3, "probability of third class")

#PX sabete
#PX given class sabet nist darimesh? ba hamoon mu_p ha


from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X, Y)
y_pred = classifier.predict(X)
cm = confusion_matrix(Y, y_pred)
print(cm)
fail = 0
fc1 = 0
fc2 = 0
failclass3 = 0
for i in range(3):
    for j in range(3):
        if i is not j:
            fail += cm[i, j]
            if j is 0:
                fc1 += cm[i, j]
            if j is 1:
                fc2 += cm[i, j]
            if j is 2:
                failclass3 += cm[i, j]
print(fail, "number of Misclassified which will be", fail/N, "% misclassified in total")
print(fc1/len(c1), "% misclassified in class 1")
print(fc2/len(c2), "% misclassified in class 2")
print(failclass3/len(c3), "% misclassified in class 3")

plt.scatter(X[c1, 0], X[c1, 1], color='yellow', label='class 1')
plt.scatter(X[c2, 0], X[c2, 1], color='red', label='class 2')
plt.scatter(X[c3, 0], X[c3, 1], color='green', label='class 3')
plt.legend
plt.show()

# if you uncomment the first comments you get the next question answered