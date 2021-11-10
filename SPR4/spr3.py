import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

mu1 = [1, 1]
mu2 = [0, 0]
sigma = 0.2
mu1 = np.array(mu1).T
mu2 = np.array(mu2).T
X1 = np.random.multivariate_normal(mu1, sigma*np.eye(2), 100)
X2 = np.random.multivariate_normal(mu2, sigma*np.eye(2), 100)


plt.scatter(X1[:, 0], X1[:, 1], c='b')
plt.scatter(X2[:, 0], X2[:, 1], c='r')
# plt.show()

# print(X1)
data = np.vstack((X1, X2))
Y_target = np.ones((200, 1))
index_C1 = range(0, X1.shape[0])
index_C2 = range(X1.shape[0], X1.shape[0]+X2.shape[0])
Y_target[index_C2] = -Y_target[index_C2]
rho = 0.01
iteration = 1000
data = np.hstack((data, np.ones(((X1.shape[0]+X2.shape[0]), 1))))

plt.scatter(X1[:, 0], X1[:, 1], c='b')
plt.scatter(X2[:, 0], X2[:, 1], c='r')

w_i = np.random.randn(3, 1) # randomize vector

# solution = False
for i in range(iteration):
    for j in range(data.shape[0]):
        x = data[j].T
        y=Y_target[j]
        delta_w = rho * x * (y - np.dot(np.reshape(x, (1,3)), w_i))
        delta_w = delta_w.T
        if np.max(np.abs(delta_w))<1.e-6:
            # solution = True
            break
        w_i += delta_w

# if solution:
#     print("get fit son")
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