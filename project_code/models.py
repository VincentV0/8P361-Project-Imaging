"""
8P361 - Project Imaging (BIA)
Code for project assignment
Group 8
Developed by:
    - Derikx, Pien
    - Donkers, Kim
    - Mulder, Eva
    - Vousten, Vincent

models.py contains functions defining the model architectures.
"""

# Reduce TensorFlow verbose (disabled)
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# Import TensorFlow and Keras libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, \
                                        GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.optimizers import SGD

# constant definitions
IMAGE_SIZE = 96

################################################################################

def CNN_01(filter1, filter2, conv_kernel=(3,3), maxpool_kernel=(4,4), \
            optimizer=SGD, lr=0.01, momentum=0.95):
    """
    DESCRIPTION: This function defines the default CNN architecture:

    Conv2D --> MaxPool2D --> Conv2D --> MaxPool2D --> Flatten --> Dense --> Dense
     ReLU                     ReLU                                ReLU     sigmoid

    Parameters:
    ----------
    TO BE ADDED

    Returns
    -------
    None
    """
    # build the model
    model = keras.Sequential()
    model.add(Conv2D(filter1, conv_kernel, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(MaxPool2D(pool_size = maxpool_kernel))
    model.add(Conv2D(filter2, conv_kernel, activation = 'relu', padding = 'same'))
    model.add(MaxPool2D(pool_size = maxpool_kernel))
    model.add(Flatten())
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))

    # compile and return the model
    model.compile(optimizer(lr=lr, momentum=momentum), loss = 'binary_crossentropy', metrics=['accuracy'])
    return model

################################################################################

def transfer_01(optimizer=SGD, lr=0.001, momentum=0.95, pretrained_weights='imagenet'):
    # build the model
    input = Input((IMAGE_SIZE, IMAGE_SIZE, 3))
    pretrained = MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights=pretrained_weights)
    # Freeze the feature-extracting layers of the pretrained model (disabled)
    #for layer in pretrained.layers:
    #    layer.trainable = False
    output = pretrained(input)
    output = GlobalAveragePooling2D()(output)
    output = Dropout(0.5)(output)
    output = Dense(1, activation='sigmoid')(output)
    model = Model(input, output)

    # compile and return the model
    model.compile(optimizer(lr=lr, momentum=momentum), loss = 'binary_crossentropy', metrics=['accuracy'])

################################################################################
