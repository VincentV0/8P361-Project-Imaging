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


def mnist_train_and_evaluate(model, batch_size=32, epochs=10, model_name=''):
    # load the dataset using the builtin Keras method
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # derive a validation set from the training set
    # the original training set is split into
    # new training set (90%) and a validation set (10%)
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

    # normalize our data values to the range [0,1]
    X_train /= 255
    X_val /= 255
    X_test /= 255

    # convert 1D class arrays to 10D class matrices
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # use this variable to name your model
    # model_name="my_first_model"

    # create a way to monitor our model in Tensorboard (disabled)
    # tensorboard = TensorBoard("logs/" + model_name)

    # train the model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_val, y_val))# , callbacks=[tensorboard])

    score = model.evaluate(X_test, y_test, verbose=0)

    #print("Model '{}' performance:".format(model_name))
    #print("Loss: ",score[0])
    #print("Accuracy: ",score[1])
    #print()

    return score[1]




model1 = Sequential()
model1.add(Flatten(input_shape=(28,28,1)))
model1.add(Dense(64, activation='relu'))
model1.add(Dense(10, activation='softmax'))
acc1 = mnist_train_and_evaluate(model1, model_name='default')

model2 = Sequential()
model2.add(Flatten(input_shape=(28,28,1)))
model2.add(Dense(96, activation='relu'))
model2.add(Dense(10, activation='softmax'))
acc2 = mnist_train_and_evaluate(model2, model_name='more neurons')

model3 = Sequential()
model3.add(Flatten(input_shape=(28,28,1)))
model3.add(Dense(64, activation='relu'))
model3.add(Dense(64, activation='relu'))
model3.add(Dense(10, activation='softmax'))
acc3 = mnist_train_and_evaluate(model3, model_name='more layers')

model4 = Sequential()
model4.add(Flatten(input_shape=(28,28,1)))
model4.add(Dense(96, activation='relu'))
model4.add(Dense(96, activation='relu'))
model4.add(Dense(10, activation='softmax'))
acc4 = mnist_train_and_evaluate(model4, model_name='more layers and neurons')

model5 = Sequential()
model5.add(Flatten(input_shape=(28,28,1)))
model5.add(Dense(10, activation='softmax'))
acc5 = mnist_train_and_evaluate(model5, model_name='no HL')

model6 = Sequential()
model6.add(Flatten(input_shape=(28,28,1)))
model6.add(Dense(64, activation='relu'))
model6.add(Dense(64, activation='relu'))
model6.add(Dense(64, activation='relu'))
model6.add(Dense(10, activation='softmax'))
acc6 = mnist_train_and_evaluate(model6, model_name='3 HL ReLU')

model7 = Sequential()
model7.add(Flatten(input_shape=(28,28,1)))
model7.add(Dense(64, activation='linear'))
model7.add(Dense(64, activation='linear'))
model7.add(Dense(64, activation='linear'))
model7.add(Dense(10, activation='softmax'))
acc7 = mnist_train_and_evaluate(model7, model_name='3 HL linear')

print('{:50}: {}'.format('Accuracy model 1 (default)',acc1))
print('{:50}: {}'.format('Accuracy model 2 (more neurons)',acc2))
print('{:50}: {}'.format('Accuracy model 3 (more layers)',acc3))
print('{:50}: {}'.format('Accuracy model 4 (more layers and neurons)',acc4))
print('{:50}: {}'.format('Accuracy model 5 (no HL)',acc5))
print('{:50}: {}'.format('Accuracy model 6 (3 HL ReLU)',acc6))
print('{:50}: {}'.format('Accuracy model 7 (3 HL linear)',acc7))
