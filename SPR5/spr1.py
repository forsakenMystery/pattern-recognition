import numpy as np
import matplotlib.pyplot as plt
# import torch
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers


m = np.array([[-5, 5, 5, -5], [5, -5, 5, -5]])
n = 100
s = 2


def data_generator(m, s, n):
    S = s*np.eye(2)
    x = []
    for i in range(m.shape[1]):
        generate = np.random.multivariate_normal(m[:, i].T, S, n)
        if len(x) is 0:
            x = generate
        else:
            x = np.vstack([x, generate])
    y = np.hstack([np.ones([1, n]), np.ones([1, n]), -np.ones([1, n]), -np.ones([1, n])])
    return x.T, y


np.random.seed(0)
x1, y1 = data_generator(m, s, n)


np.random.seed(10)
x2, y2 = data_generator(m, s, n)

s = 5

np.random.seed(0)
x3, y3 = data_generator(m, s, n)


np.random.seed(10)
x4, y4 = data_generator(m, s, n)


def animate(x, y, title):
    c1 = x[:, :200]
    c2 = x[:, 200:]
    plt.scatter(c1[0, :], c1[1, :], c='b', label='Class 1')
    plt.scatter(c2[0, :], c2[1, :], c='r', label='Class 2')
    plt.legend()
    plt.title(title)
    plt.show()


animate(x1, y1, "first data set")
animate(x2, y2, "second data set")
animate(x3, y3, "third data set")
animate(x4, y4, "fourth data set")
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.svm import SVC

C=[1, 100, 1000]
gamma=[.5, 1, 2, 4]

for i in range(len(C)):
    for j in range(len(gamma)):
        c = C[i]
        g = gamma[j]
        svclassifier = SVC(kernel='rbf', gamma=g, tol=.001, C=c)
        svclassifier.fit(x1.T, y1.ravel())
        y_pred = svclassifier.predict(x2.T)
        print(confusion_matrix(y2.ravel(), y_pred))
        print(classification_report(y2.ravel(), y_pred))

        plot_decision_regions(x1.T, y1.astype(np.integer).ravel(), clf=svclassifier, legend=2)
        plt.title(f"first data set, SVM on train C=%d gamma=%.2f" % (c, g))
        plt.show()

        plot_decision_regions(x2.T, y2.astype(np.integer).ravel(), clf=svclassifier, legend=2)
        plt.title(f"first data set, SVM on test C=%d gamma=%.2f" % (c, g))
        plt.show()

        svclassifier = SVC(kernel='rbf', gamma=g, tol=.001, C=c)
        svclassifier.fit(x3.T, y3.ravel())
        y_pred = svclassifier.predict(x4.T)
        print(confusion_matrix(y4.ravel(), y_pred))
        print(classification_report(y4.ravel(), y_pred))

        plot_decision_regions(x3.T, y3.astype(np.integer).ravel(), clf=svclassifier, legend=2)
        plt.title(f"second data set, SVM on train C=%d gamma=%.2f" % (c, g))
        plt.show()

        plot_decision_regions(x4.T, y4.astype(np.integer).ravel(), clf=svclassifier, legend=2)
        plt.title(f"second data set, SVM on test C=%d gamma=%.2f" % (c, g))
        plt.show()


# ====================================================================

# K(x,xi) = exp(-gamma * sum((x – xi)^2)

# class Net(torch.nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(Net, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
#         self.tanh = torch.nn.Tanh()
#         self.fc2 = torch.nn.Linear(self.hidden_size, 1)
#         self.sigmoid = torch.nn.Sigmoid()
#
#     def forward(self, x):
#         hidden = self.fc1(x)
#         tanh = self.tanh(hidden)
#         output = self.fc2(tanh)
#         output = self.sigmoid(output)
#         return output
#
#
# model = Net(2, 4)
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#
#
# x_train = torch.FloatTensor(x1.T)
# y_train = torch.FloatTensor(y1)
#
# x_test = torch.FloatTensor(x2.T)
# y_test = torch.FloatTensor(y2)
#
#
# model.eval()
# y_pred = model(x_test)
# before_train = criterion(y_pred.squeeze(), y_test)
# print('Test loss before training', before_train.item())
#
# model.train()
# epochs = 1000
# for epoch in range(epochs):
#     optimizer.zero_grad()
#     y_pred = model(x_train)
#     loss = criterion(y_pred.squeeze(), y_train)
#     if epoch ==1 or epoch == 999:
#         print(y_pred)
#     print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
#     loss.backward()
#     optimizer.step()
#
# model.eval()
# y_pred = model(x_test)
# after_train = criterion(y_pred.squeeze(), y_test)
# print('Test loss after Training', after_train.item())
#
#
# def plot_decision_region(model, train_X, train_Y, test_X, test_Y, resolution=False, lh=0, rh=0, lv=0, rv=0, lr=0.1, rr=0.1):
#     if not resolution:
#         x1range = np.arange(min(train_X[:, 0]), max(train_X[:, 0]), 0.1)
#         x2range = np.arange(min(train_X[:, 1]), max(train_X[:, 1]), 0.1)
#         grid = np.array(np.meshgrid(x1range, x2range))
#         print(grid.shape)
#         model.eval()
#         grid = torch.FloatTensor(grid.T)
#         y_pred = model(grid)
#         boo = torch.where(y_pred >=0.5, torch.ones(1), torch.zeros(1))
#         print(boo)
#         M = np.zeros((x1range.shape[0], x2range.shape[0], 3))
#         print(M.shape)
#         print(boo.shape)
#         print(x1range.shape)
#         for i in range(boo.shape[0]):
#             for j in range(boo.shape[1]):
#                 M[i, j] = [0, 0, 255]
#                 if boo[i, j]==1:
#                     M[i,j, 0] = 255
#                     M[i,j, 2] = 255
#         plt.imshow(M)
#         plt.show()
#         # to do: fix this thing later I think it ain't work
#
#
# plot_decision_region(model, x1.T, y1, x2.T, y2)

# model = keras.Sequential()
# model.add(keras.Input(shape=(2,)))
# model.add(layers.Dense(2, activation="tanh"))
# model.add(layers.Dense(1, activation="sigmoid"))
# model.summary()