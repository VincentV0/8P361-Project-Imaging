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

from cnn import get_pcam_generators

# constant definitions
IMAGE_SIZE = 96
DATA_PATH  = '../../data/';
MODEL_NAME = 'CNN_01';

# Open the model, compile and load its weights
json_file = open("../CNN results/" + MODEL_NAME + ".json", 'r')
json_read = json_file.read()
json_file.close()
model = keras.models.model_from_json(json_read);
model.compile(SGD(lr=0.01, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])
model.load_weights("../CNN results/" + MODEL_NAME + "_weights.hdf5");

# Load the data
train_gen, val_gen = get_pcam_generators(DATA_PATH)

# Calculate the accuracy and loss using model.evaluate
loss, accur = model.evaluate(val_gen)

# Predict the validation samples using model.predict and binarize the predictions
y_pred = model.predict(val_gen,verbose = 0)
y_pred_bin = (y_pred > 0.5).astype(int)

# Calculate scores
fpr, tpr, _ = roc_curve(val_gen.labels, y_pred)
auc_score = roc_auc_score(val_gen.labels, y_pred)
f1 = f1_score(val_gen.labels, y_pred_bin)
acc = accuracy_score(val_gen.labels, y_pred_bin)

print("Loss     (model.evaluate) = {}".format(loss))
print("Accuracy (model.evaluate) = {}".format(accur))

print("AUC       (model.predict) = {}".format(auc_score))
print("F1        (model.predict) = {}".format(f1))
print("Accuracy  (model.predict) = {}".format(acc))
