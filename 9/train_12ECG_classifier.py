#!/usr/bin/env python
from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)
import numpy as np, os, sys
from scipy.io import loadmat

import keras
import random
import network

from read_main import read_data
from sklearn.model_selection import train_test_split



def train_12ECG_classifier(input_directory, output_directory):

    MAX_EPOCHS = 50
    
    model = network.build_network()    
         
    stopping = keras.callbacks.EarlyStopping(patience=5)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        factor=0.1,
        patience=3,
        min_lr= 0.001*0.001)
    
    filepath = os.path.join(output_directory, "model.h5")#-{epoch:02d}-{val_accuracy:4f}-{val_loss:.4f}.h5")
         
    checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
         
    train_x, train_y, dev_x, dev_y = read_data(input_directory)
         
         
    model.fit(
        train_x, train_y,
        batch_size=12,
        epochs=MAX_EPOCHS,
        validation_data=(dev_x, dev_y),
        callbacks= [checkpoint, reduce_lr,stopping])
         
    #log_file = os.path.join(output_directory, "stdout")
        
    #print("Logging to {}".format(log_file))
    #sys.stdout = Logger(log_file)
