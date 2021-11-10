import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

mu1 = [1, 1]
mu2 = [0, 0]
sigma = 0.2
mu1 = np.array(mu1).T
mu2 = np.array(mu2).T
X1_1 = np.array([])
while X1_1.shape[0]<50:#while data is not ready
    X1 = np.random.multivariate_normal(mu1, sigma*np.eye(2), 1000)
    b = np.where(X1[:, 0]+X1[:, 1]<1)
    X1_1 = X1[b]

X2_1 = np.array([])
while X2_1.shape[0]<50:#while data is not ready
    X2 = np.random.multivariate_normal(mu2, sigma*np.eye(2), 1000)
    b = np.where(X2[:, 0]+X2[:, 1]>1)
    X2_1 = X2[b]

X1_1 = X1_1[np.random.choice(X1_1.shape[0], 50)]
# print(X1_1)
X2_1 = X2_1[np.random.choice(X2_1.shape[0], 50)]
# print("=======================================")
# print("=======================================")
# print(X2_1)

X1 = X1_1
X2 = X2_1
plt.scatter(X1[:, 0], X1[:, 1], c='b')
plt.scatter(X2[:, 0], X2[:, 1], c='r')
# plt.show()

# print(X1)
data = np.vstack((X1, X2))
# print(data)
# print(X1.shape)
# print(np.ones(((X1.shape[0]+X2.shape[0]), 1)))
data = np.hstack((data, np.ones(((X1.shape[0]+X2.shape[0]), 1))))
# print(data)
index_C1 = range(0, X1.shape[0])
index_C2 = range(X1.shape[0], X1.shape[0]+X2.shape[0])
# print(index_C2[0])
data[index_C2, :] = -data[index_C2, :]# negation to simplify computation instead of t*w_i*x we combine x and t
# print(data)

rho = 0.7
iteration = 1000

w_i = np.array([[0], [0], [0]])
print("=====================")
print("start training")
for i in range(iteration): # give a cap so if there is no linear we break out
    error = 0
    for j in range(data.shape[0]):
        # look at each sample to classify all
        # print(data[j, :])
        # print(w_i)
        # print(data[j, :]*w_i)
        # print(np.dot(data[j, :], w_i))
        # print(np.multiply(data[j, :], w_i))
        datum = data[j, :]
        if np.dot(datum, w_i)[0] <= 0:
            error += 1
            # print(w_i.shape)
            # print(datum.shape)
            transpose = datum.T
            w_i = w_i + (rho*np.reshape(transpose, (transpose.shape[0], 1)))
            print(w_i)
    if error is 0: # no more error
        break

y_intercept = (-w_i[2] / w_i[0], 0)
print(y_intercept)
x_intercept = (0, -w_i[2] / w_i[1])
print(x_intercept)


def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin, xmax], [ymin, ymax])
    ax.add_line(l)
    return l


newline(y_intercept, x_intercept)
plt.show()
