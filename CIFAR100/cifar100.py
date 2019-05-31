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
y_train_one_hot = keras.utils.to_categorical(y_train)
y_test_one_hot = keras.utils.to_categorical(y_test)

# ? splitting the data into training and validation dataset
x_train, x_valid, x_label, valid_label = train_test_split(
    x_train, y_train_one_hot, test_size=0.2, random_state=13)

batch_size = 128  # 64 and 256 is also preferable
epochs = 20  # Number of Epochs
num_classes = 100  # Number of Output Classes

#! NETWORK
"""
Overfitting is one of the biggest problems we face. Luckily, Dropout is a technique that addresses this problem.
During the training process, it randomly drops units from the neural network.
(More about overfitting can be found here http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
Batch Normalization normalizes the activation of the previous layer at each batch
In other workds, it applies a transformation that maintains the mean activation close to 0 
and the activation standard deviation close to 1.
(More about Batch Normalization can be found here https://arxiv.org/abs/1502.03167)
"""

cifar100model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='linear',
           padding='same', input_shape=x_train.shape[1:]),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Conv2D(32, kernel_size=(3, 3), activation='linear',
           padding='same', input_shape=x_train.shape[1:]),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    MaxPooling2D((2, 2), padding='same'),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='linear', padding='same'),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='linear', padding='same'),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='linear', padding='same'),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='linear', padding='same'),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2), padding='same'),
    Dropout(0.4),
    Flatten(),
    Dense(128, activation='linear'),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

#! Compiling the model
cifar100model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=Adam(), metrics=['accuracy'])
cifar100model.summary()

# fit() will return a history object
cifar100_train = cifar100model.fit(x_train, x_label, batch_size=batch_size,
                                   epochs=epochs, verbose=1, validation_data=(x_valid, valid_label))


#! TESTING
test_eval = cifar100model.evaluate(x_test, y_test_one_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

accuracy = cifar100_train.history['acc']
val_accuracy = cifar100_train.history['val_acc']
loss = cifar100_train.history['loss']
val_loss = cifar100_train.history['val_loss']
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

cifar100model.save("cifar100modeldropout.h5py")

#!PREDICTION
predicted_classes = cifar100model.predict(x_test)
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
correct = np.where(predicted_classes == y_test)[0]
print("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_test[correct].reshape(28, 28),
               cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(
        predicted_classes[correct], y_test[correct]))
    plt.tight_layout()

incorrect = np.where(predicted_classes != y_test)[0]
print("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_test[incorrect].reshape(28, 28),
               cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(
        predicted_classes[incorrect], y_test[incorrect]))
    plt.tight_layout()

target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_test, predicted_classes, target_names=target_names))
