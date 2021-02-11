'''
TU/e BME Project Imaging 2021
Convolutional neural network for PCAM
Author: Suzanne Wetstein
'''

# disable overly verbose tensorflow logging
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt

# constant definitions
IMAGE_SIZE = 96
DATA_PATH  = '../../data/';

################################ ↓ MODELS ↓ ####################################
def get_model_1(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64):
     # build the model
     model = keras.Sequential()
     model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
     model.add(MaxPool2D(pool_size = pool_size))
     model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))
     model.add(Flatten())
     model.add(Dense(64, activation = 'relu'))
     model.add(Dense(1, activation = 'sigmoid'))

     # compile and return the model
     model.compile(SGD(lr=0.01, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])
     return model

def get_model_2(kernel_size=(3,3), pool_size=(4,4), first_filters=8, second_filters=4):
     # build the model
    model = keras.Sequential()
    model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(MaxPool2D(pool_size = pool_size))
    model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
    model.add(MaxPool2D(pool_size = pool_size))
    model.add(Conv2D(2, kernel_size, activation = 'relu', padding = 'same'))
    model.add(MaxPool2D(pool_size = pool_size))
    model.add(Conv2D(1, (1,1), activation = 'sigmoid', padding = 'same'))
    model.add(Flatten())
    model.summary()

    # compile and return the model
    model.compile(SGD(lr=0.01, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])
    return model

################################ ↑ MODELS ↑ ####################################


################################################################################

def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):
    """
    DESCRIPTION: This function imports the data and divides it into batches.
    """
    # dataset parameters
    train_path = os.path.join(base_dir, 'train')
    valid_path = os.path.join(base_dir, 'valid')
    RESCALING_FACTOR = 1./255

    # instantiate data generators
    datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

    train_gen = datagen.flow_from_directory(train_path,
                                         target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                         batch_size=train_batch_size,
                                         class_mode='binary')

    val_gen = datagen.flow_from_directory(valid_path,
                                         target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                         batch_size=val_batch_size,
                                         class_mode='binary')

    return train_gen, val_gen

################################################################################

def plot_ROC_curve(fpr, tpr, folder, model_name):
    """
    DESCRIPTION: This function plots the ROC curves of the different runs
    and saves it to a png file.

    Parameters:
    ----------
    fpr: the list with the false positive rates for every run
    tpr: the list with the true positive rates for every run
    folder: the folder to save to
    model_name: the name to be included in the filename.

    Returns
    -------
    None
    """
    # Make the figure
    fig,ax = plt.subplots()

    # Plot the ROC curves for every run and every fold
    ax.plot(fpr, tpr, color='#ffaa75',LineWidth=1)

    # Style plot and save to file
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    fig.savefig(os.path.join(folder, model_name + "_ROC_curve"))

################################################################################

def train_and_evaluate_CNN(model, model_name='CNN', save_folder='./'):
    """
    DESCRIPTION: This function trains and evaluates the CNN.
    """
    # get the data generators
    train_gen, val_gen = get_pcam_generators(DATA_PATH)

    # save the model and weights
    model_filepath = os.path.join(save_folder, model_name + '.json');
    weights_filepath = os.path.join(save_folder, model_name + '_weights.hdf5');

    model_json = model.to_json() # serialize model to JSON
    with open(model_filepath, 'w') as json_file:
        json_file.write(model_json)

    # define the model checkpoint and Tensorboard callbacks
    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard = TensorBoard(os.path.join('../TB_logs', model_name))
    callbacks_list = [checkpoint, tensorboard]

    # train the model
    train_steps = train_gen.n//train_gen.batch_size
    val_steps = val_gen.n//val_gen.batch_size

    history = model.fit(train_gen, steps_per_epoch=train_steps,
                        validation_data=val_gen,
                        validation_steps=val_steps,
                        epochs=3,
                        callbacks=callbacks_list)

    # evaluate model
    y_pred = model.predict(val_gen,verbose = 0)
    y_pred_bin = (y_pred > 0.5).astype(int)
    loss, acc_ev = model.evaluate(X_test, y_test, verbose=0)

    # calculate scores
    fpr, tpr, _ = roc_curve(val_gen.labels, y_pred)
    auc_score = roc_auc_score(val_gen.labels, y_pred)
    f1 = f1_score(val_gen.labels, y_pred_bin)
    acc = accuracy_score(val_gen.labels, y_pred_bin)

    # save results
    plot_ROC_curve(fpr,tpr, save_folder, model_name)
    with open(os.path.join(save_folder, model_name + '_scores.txt'), 'w') as result_file:
        result_file.write('AUC score      = {}\n'.format(auc_score))
        result_file.write('F1 score       = {}\n'.format(f1))
        result_file.write('Accuracy score = {}\n'.format(acc))
        result_file.write('Accuracy (model.evaluate) = {}'.format(acc_ev))

    return model

################################################################################

# get the 1st model
#model1 = get_model_1()
#trained_model1 = train_and_evaluate_CNN(model1, model_name='CNN_01', save_folder='../CNN results/')

# get the 2nd model
#model2 = get_model_2()
#trained_model2 = train_and_evaluate_CNN(model2, model_name='CNN_02', save_folder='../CNN results/')
