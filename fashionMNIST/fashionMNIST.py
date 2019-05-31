# WRITTEN BY Gansaikhan Shur ~Khan
# The Following Code uses CNN to classify Fashion MNIST Dataset

from tensorflow import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

fmnistData = keras.datasets.fashion_mnist

#! Loading the Fashion MNIST Dataset
(train_X, train_Y), (test_X, test_Y) = fmnistData.load_data()

# print('Training data shape : ', train_X.shape, train_Y.shape)
# print('Testing data shape : ', test_X.shape, test_Y.shape)
# plt.figure(figsize=[5, 5])
# * Display the first image in training data
# plt.subplot(121)
# plt.imshow(train_X[0], cmap='gray')
# plt.title("Ground Truth : {}".format(train_Y[0]))
# * Display the first image in testing data
# plt.subplot(122)
# plt.imshow(test_X[0], cmap='gray')
# plt.title("Ground Truth : {}".format(test_Y[0]))
# plt.show()

# ? convert each 28 x 28 image of the train and test set into a matrix of size 28 x 28 x 1
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)

# ? images are grayscale images have pixel values that range from 0 to 255 (28x28 pixel)
# ? rescale the pixel values in range 0 - 1 inclusive
train_X = (train_X.astype('float32')) / 255.0
test_X = (test_X.astype('float32')) / 255.0

# ? convert the training and testing labels from categorical to one-hot encoding vectors
train_Y_one_hot = keras.utils.to_categorical(train_Y)
test_Y_one_hot = keras.utils.to_categorical(test_Y)

# ('Original label:', 9)
# ('After conversion to one-hot:', array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]))

train_X, valid_X, train_label, valid_label = train_test_split(
    train_X, train_Y_one_hot, test_size=0.2, random_state=13)

#! NETWORK

# * Input (28x28x1) ==> Convolution1 32-3x3 filters ==> Max Pooling 2x2 ==>
#                     Convolution2 64-3x3 filters ==> Max Pooling 2x2 ==>
#                     Convolution3 128-3x3 filters ==> Max Pooling 2x2 ==>
#                   Flatten ==> Dense Layer 128 Nodes ==> Output Layer 10 Nodes

batch_size = 64  # 128 and 256 is also preferable
epochs = 25  # Number of Epochs
num_classes = 10  # Number of Output Classes

# * Basic Ideas Behind Layers that are to be used!!!
# Conv2D() Because We're working with images
# Leaky ReLU activation function to help the network learn non-linearity
# 10 different classes need a non-linear decision boundary that could separate these ten classes which are not linearly separable.
# during the training, ReLU units can "die". This can happen when a large gradient flows through a ReLU neuron: it can cause the weights to update in such a way that the neuron will never activate on any data point again.
# Leaky ReLUs attempt to fix the problem of dying Rectified Linear Units (ReLUs)
# then MaxPooling2D() and so on until we reach to the last Dense Layers

fashion_model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='linear',
           padding='same', input_shape=(28, 28, 1)),
    LeakyReLU(alpha=0.1),
    MaxPooling2D((2, 2), padding='same'),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='linear', padding='same'),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='linear', padding='same'),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    Dropout(0.4),
    Flatten(),
    Dense(128, activation='linear'),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

#! Compiling the model
fashion_model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=Adam(), metrics=['accuracy'])
fashion_model.summary()

# fit() will return a history object
fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,
                                  epochs=epochs, verbose=1, validation_data=(valid_X, valid_label))

#! TESTING

test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

accuracy = fashion_train.history['acc']
val_accuracy = fashion_train.history['val_acc']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

fashion_model.save("fashion_model_dropout.h5py")

#! PREDICTION
predicted_classes = fashion_model.predict(test_X)
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
correct = np.where(predicted_classes == test_Y)[0]
print("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3, 3, i+1)
    plt.imshow(test_X[correct].reshape(28, 28),
               cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(
        predicted_classes[correct], test_Y[correct]))
    plt.tight_layout()

incorrect = np.where(predicted_classes != test_Y)[0]
print("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3, 3, i+1)
    plt.imshow(test_X[incorrect].reshape(28, 28),
               cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(
        predicted_classes[incorrect], test_Y[incorrect]))
    plt.tight_layout()

target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_Y, predicted_classes, target_names=target_names))
