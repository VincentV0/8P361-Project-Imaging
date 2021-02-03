"""
TU/e BME Project Imaging 2021
Simple multiLayer perceptron code for MNIST
Author: Suzanne Wetstein
"""

# disable overly verbose tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf


# import required packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard


def mnist_train_and_evaluate_4_class(model, batch_size=32, epochs=10, model_name=''):
    # load the dataset using the builtin Keras method
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # derive a validation set from the training set
    # the original training set is split into
    # new training set (90%) and a validation set (10%)

    # Other experiment:
    y_train_new = y_train.copy()
    y_train_new[[(y_train == 1) | (y_train == 7)]] = 0
    y_train_new[[(y_train == 0) | (y_train == 6) | (y_train == 8) | (y_train == 9)]] = 1
    y_train_new[[(y_train == 2) | (y_train == 5)]] = 2
    y_train_new[[(y_train == 3) | (y_train == 4)]] = 3
    y_test_new = y_test.copy()
    y_test_new[[(y_test == 1) | (y_test == 7)]] = 0
    y_test_new[[(y_test == 0) | (y_test == 6) | (y_test == 8) | (y_test == 9)]] = 1
    y_test_new[[(y_test == 2) | (y_test == 5)]] = 2
    y_test_new[[(y_test == 3) | (y_test == 4)]] = 3
    y_test = y_test_new;
    y_train = y_train_new;


    X_train, X_val = train_test_split(X_train, test_size=0.10, random_state=101)
    y_train, y_val = train_test_split(y_train, test_size=0.10, random_state=101)

    # the shape of the data matrix is NxHxW, where
    # N is the number of images,
    # H and W are the height and width of the images
    # keras expect the data to have shape NxHxWxC, where
    # C is the channel dimension
    X_train = np.reshape(X_train, (-1,28,28,1))
    X_val = np.reshape(X_val, (-1,28,28,1))
    X_test = np.reshape(X_test, (-1,28,28,1))

    # convert the datatype to float32
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')

    # n|malize our data values to the range [0,1]
    X_train /= 255
    X_val /= 255
    X_test /= 255

    # convert 1D class arrays to 10D class matrices
    y_train = to_categorical(y_train, 4)
    y_val = to_categorical(y_val, 4)
    y_test = to_categorical(y_test, 4)

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # use this variable to name your model
    # model_name="my_first_model"

    # create a way to monitor our model in Tensorboard (disabled)
    # tensorboard = TensorBoard("logs/" + model_name)

    # train the model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_val, y_val))# , callbacks=[tensorboard])

    # evaluate model and return accuracy score
    score = model.evaluate(X_test, y_test, verbose=0)
    return score[1]




model8 = Sequential()
model8.add(Flatten(input_shape=(28,28,1)))
model8.add(Dense(64, activation='relu'))
model8.add(Dense(4, activation='softmax'))
acc8 = mnist_train_and_evaluate_4_class(model8, model_name='default')



print('{:50}: {}'.format('Accuracy model 8 (default)',acc8))
