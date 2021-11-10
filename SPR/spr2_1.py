import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def generate_data(mean, covariance, num=1):
    return np.random.multivariate_normal(mean, covariance, num)


cov = [[0.2]]
mu1 = [0]
mu2 = [2]
N = 1000


def sample_data(num=N):
    sampled = []
    y_sampled = []
    class1 = []
    class2 = []
    for i in range(num):
        prob = np.random.randint(0, 99)
        if prob < 33:
            sampled.append(generate_data(mu1, cov)[0])
            y_sampled.append(0)
            class1.append(i)
        else:
            sampled.append(generate_data(mu2, cov)[0])
            y_sampled.append(1)
            class2.append(i)
    return np.array(sampled), np.array(y_sampled), class1, class2


X, Y, c1, c2 = sample_data()
h = 0.01

x = np.arange(-5, 5, h)
plt.plot(x, ((1/3)*norm.pdf(x, loc=mu1[0], scale=0.2))+((2/3)*norm.pdf(x, loc=mu2[0], scale=0.2)))


def parzen(X, h=h, left=-5, right=5, N=N):
    step = h
    k = 0
    x = left
    px = []
    while x < right + step/2:
        px.append(0)
        for i in range(N):
            xi = X[i]
            px[k] += np.exp(-(x-xi)*(x-xi)/(2*h**2))
        px[k] *= (1/N)*(1/(((2*np.pi)**(.5))*(h)))
        k += 1
        x += step
    return np.array(px)


px = parzen(X)
px = px[0:x.size, :]

print(px.shape)
print(x.size)
plt.scatter(x, px)
plt.show()

