import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense , merge
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical
from keras.utils.vis_utils import model_to_dot
from keras.callbacks import ReduceLROnPlateau
from keras.layers.merge import dot
from keras.models import Model
from keras.layers import Dropout, Flatten,Activation,Input,Embedding
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
epochs=20
NUM_SAMPLES = 50
x = np.random.uniform(0., 1., 10000)
x = np.sort(x, axis=0)
sg = SGD(lr=1e-3)
y = (0.3 + .2*np.cos(2*np.pi*x)).ravel()


x_test = np.random.uniform(0., 1., NUM_SAMPLES)
x_test = np.sort(x_test, axis=0)

number = Input(shape=(1,),name='input')
nn_inp=Dense(2, activation='tanh')(number)
nn_inp=Dense(2, activation='sigmoid')(nn_inp)
nn_inp=Dense(1)(nn_inp)
nn_model =keras.models.Model(number, nn_inp)
nn_model.summary()
nn_model.compile(optimizer=sg,loss='mse')
History = nn_model.fit(x, y, epochs=epochs, verbose=1)
y_test = nn_model.predict(x_test)
print(y_test)
plt.plot(x_test, y_test, c='r', label='predicting test _ 2 hidden layer with both has 2 node')
plt.plot(x, y, c='b', label='the true figure')
plt.legend()
plt.show()

number = Input(shape=(1,),name='input')
nn_inp=Dense(2, activation='relu')(number)
nn_inp=Dense(3, activation='tanh')(nn_inp)
nn_inp=Dense(2, activation='sigmoid')(nn_inp)
nn_inp=Dense(1)(nn_inp)
nn_model =keras.models.Model(number, nn_inp)
nn_model.summary()
nn_model.compile(optimizer=sg,loss='mse')
History = nn_model.fit(x, y, epochs=epochs, verbose=1)
y_test = nn_model.predict(x_test)
print(y_test)
plt.plot(x_test, y_test, c='r', label='predicting test _ 3 hidden layer 2, 3, and 2 nodes respectively')
plt.plot(x, y, c='b', label='the true figure')
plt.legend()
plt.show()

number = Input(shape=(1,),name='input')
nn_inp=Dense(4, activation='relu')(number)
nn_inp=Dense(2, activation='tanh')(nn_inp)
nn_inp=Dense(4, activation='sigmoid')(nn_inp)
nn_inp=Dense(1)(nn_inp)
nn_model =keras.models.Model(number, nn_inp)
nn_model.summary()
nn_model.compile(optimizer=sg,loss='mse')
History = nn_model.fit(x, y, epochs=epochs, verbose=1)
y_test = nn_model.predict(x_test)
print(y_test)
plt.plot(x_test, y_test, c='r', label='predicting test _ 3 hidden layer 4, 2, and 4 nodes respectively')
plt.plot(x, y, c='b', label='the true figure')
plt.legend()
plt.show()



number = Input(shape=(1,),name='input')
nn_inp=Dense(4, activation='tanh')(number)
nn_inp=Dense(8, activation='tanh')(nn_inp)
nn_inp=Dense(4, activation='relu')(nn_inp)
nn_inp=Dense(2, activation='sigmoid')(nn_inp)
nn_inp=Dense(1)(nn_inp)
nn_model =keras.models.Model(number, nn_inp)
nn_model.summary()
nn_model.compile(optimizer=sg,loss='mse')
History = nn_model.fit(x, y, epochs=epochs, verbose=1)
y_test = nn_model.predict(x_test)
print(y_test)
plt.plot(x_test, y_test, c='r', label='predicting test _ 4 hidden layer 4, 8, 4, and 2 nodes respectively')
plt.plot(x, y, c='b', label='the true figure')
plt.legend()
plt.show()