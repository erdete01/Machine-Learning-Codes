# MNIST is a classic 'Hello World' example in Machine Learning
# Written by Gansaikhan Shur
# (the original dataset can be found here http://yann.lecun.com/exdb/mnist/)

from tensorflow import keras
from keras.datasets import mnist

mnistDataset = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnistDataset.load_data()
