# Written by Gansaikhan
# The Following Code uses CNN to classify CIFAR 100 Dataset
# (original dataset can be found here at https://www.cs.toronto.edu/~kriz/cifar.html)

from tensorflow import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from matplotlib.pyplot import imread
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

cifar100 = keras.datasets.cifar100
# !Loading the CIFAR100 Dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

print('Training data shape : ', x_train.shape, y_train.shape)
print('Testing data shape : ', x_test.shape, y_test.shape)
#plt.figure(figsize=[5, 5])
# * Display the first image in training data
# plt.subplot(121)
#plt.imshow(x_train[0], cmap='gray')
#plt.title("Label : {}".format(y_train[0]))
# * Display the first image in testing data
# plt.subplot(122)
#plt.imshow(x_test[0], cmap='gray')
#plt.title("Label : {}".format(y_test[0]))
# plt.show()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# ? from categorical to one-hot encoding vectors
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
