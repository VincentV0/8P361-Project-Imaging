"""
8P361 - Project Imaging (BIA)
Code for project assignment
Group 8
Developed by:
    - Derikx, Pien
    - Donkers, Kim
    - Mulder, Eva
    - Vousten, Vincent

main.py contains the code for training and evaluating the model.
"""


# Reduce TensorFlow verbose (disabled)
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# Load Tensorflow and Keras libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import SGD

# Load Scipy, Numpy and Matplotlib libraries
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Load functions from other files
from utils import plot_ROC_curve, get_generators, save_history
import models

# constant definitions
IMAGE_SIZE  = 96
DATA_PATH   = '../../data/';
SAVE_PATH   = '../results/';
TB_LOG_PATH = '../TB_logs';
################################################################################

def train_and_evaluate(model, model_name='CNN', save_folder='./', nr_epochs=10, \
                        pred_threshold=0.5):
    """
    DESCRIPTION: This function trains and evaluates the CNN.

    Parameters:
    ----------
    model:       A fully constructed and compiled model;
    model_name:  Used for the filenames of results and the TensorBoard logs
    save_folder: Used to determine where results should be saved (excl. Tensorboard
                 logs!)

    Returns:
    -------
    model:       The fitted model is returned.
    """

    # Get the data generators
    train_gen, val_gen, val_gen_no_shuffle = get_generators(DATA_PATH)

    # Define filepaths to save the model and weights
    model_filepath = os.path.join(save_folder, model_name + '.json');
    weights_filepath = os.path.join(save_folder, model_name + '_weights.hdf5');

    # Save the model to a .json file
    model_json = model.to_json()
    with open(model_filepath, 'w') as json_file:
        json_file.write(model_json)

    # Define the model checkpoint and TensorBoard callbacks
    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    tensorboard = TensorBoard(os.path.join(TB_LOG_PATH, model_name))
    callbacks_list = [checkpoint, tensorboard]

    # Define the number of training samples to use during a single epoch, and
    # the number of validation samples validated on per epoch.
    train_steps = train_gen.n//train_gen.batch_size
    val_steps = val_gen.n//val_gen.batch_size

    # Train the model
    history = model.fit(train_gen, steps_per_epoch=train_steps,
                        validation_data=val_gen,
                        validation_steps=val_steps,
                        epochs=nr_epochs,
                        callbacks=callbacks_list)

    # Evaluate model
    y_pred = model.predict(val_gen_no_shuffle,verbose = 0)
    y_pred_bin = (y_pred > pred_threshold).astype(int)

    # Calculate scores
    fpr, tpr, _ = roc_curve(val_gen.labels, y_pred)
    auc_score = roc_auc_score(val_gen.labels, y_pred)
    f1 = f1_score(val_gen.labels, y_pred_bin)
    acc = accuracy_score(val_gen.labels, y_pred_bin)

    # Save results
    save_history(history, model_name, save_folder)
    plot_ROC_curve(fpr,tpr, save_folder, model_name)
    with open(os.path.join(save_folder, model_name + '_scores.txt'), 'w') as result_file:
        result_file.write('AUC score      = {}\n'.format(auc_score))
        result_file.write('F1 score       = {}\n'.format(f1))
        result_file.write('Accuracy score = {}\n'.format(acc))
    return model

################################################################################

if __name__ == "__main__":
    model1 = models.CNN_01(32, 64, SGD, lr=0.01, momentum=0.95)  # default
    train_and_evaluate(model1, 'model_01', save_folder = SAVE_PATH)
