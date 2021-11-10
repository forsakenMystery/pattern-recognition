import numpy as np
import matplotlib.pyplot as plt
X1 = [[0, 0], [0, 1]]
X2 = [[1, 0], [1, 1]]

X1 = np.array(X1)
X2 = np.array(X2)

plt.scatter(X1[:, 0], X1[:, 1], c='b')
plt.scatter(X2[:, 0], X2[:, 1], c='r')

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

rho = 1
iteration = 10

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
xc = .5
plt.axvline(x=xc, label='decision line at x = {}'.format(xc), c='g')
plt.show()
y_intercept = (-w_i[2] / w_i[0], 0)
print(y_intercept)
# x_intercept = (0, -w_i[2] / w_i[1]) it's a vertical line intercept x is infinity so nvm
