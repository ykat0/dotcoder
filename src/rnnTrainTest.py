#!/usr/bin/env python

# rnnTrainTest.py ver. 1.0.0
# Copyright (C) 2018 Yuki Kato
# This script is used to train/test a recurrent neural network (RNN) for predicting if given transcript sequences can form RNA G-quadruplexes (rG4s).
# A trained model will be written to a specified file (e.g. model.h5).
# Note that there is a limit on the matrix row size used for both training and test, which can be specified as an option.
# This script needs "keras," "tensorflow" and "h5py" to be installed.
# Usage: Type "./rnnTrainTest.py -h" in your terminal.

import argparse

parser = argparse.ArgumentParser(
    usage = './rnnTrainTest.py [options]* <data> <model>',
    description = 'This script is used to train/test a recurrent neural network (RNN) for predicting if given transcript sequences can form RNA G-quadruplexes (rG4s). A trained model will be written to a specified file (e.g. model.h5). Note that there is a limit on the matrix row size used for both training and test, which can be specified as an option. This script needs "keras," "tensorflow" and "h5py" to be installed.',
    epilog = 'rnnTrainTest.py ver. 1.0.0')
parser.add_argument('data', metavar = 'data <STRING>', type = str,
                    help = 'path to the training/test data (sequence, label) in gzipped TSV format')
parser.add_argument('model', metavar = 'model <STRING>', type = str,
                    help = 'path to the trained model in HDF5 format')
parser.add_argument('-v', '--verbose', action = "store_true",
                    help = 'show details (default: off)')
parser.add_argument('-t', '--train', action = "store_true",
                    help = 'training mode (default: off)')
parser.add_argument('-u', '--unit', metavar = '<INT>', type = int,
                    help = 'number of hidden units in both directions (default <INT> = 512)',
                    default = 512)
parser.add_argument('-d', '--dropout', metavar = '<FLOAT>', type = float,
                    help = 'dropout rate (default <FLOAT> = 0.5)',
                    default = 0.5)
parser.add_argument('-b', '--batch', metavar = '<INT>', type = int,
                    help = 'batch size (default <INT> = 64)',
                    default = 64)
parser.add_argument('-e', '--epoch', metavar = '<INT>', type = int,
                    help = 'max epochs (default <INT> = 50)',
                    default = 50)
parser.add_argument('-M', '--matrix', metavar = '<INT>', type = int,
                    help = 'max matrix row size (default <INT> = 200)',
                    default = 200)
args = parser.parse_args()

import re
import gzip
import sys

def seq2mat(tensor, seq):
    for i in range(len(seq)):
        row = [0, 0, 0, 0]
        
        if seq[i].upper() == "A":
            row[0] = 1

        elif seq[i].upper() == "C":
            row[1] = 1

        elif seq[i].upper() == "G":
            row[2] = 1

        elif seq[i].upper() == "T":
            row[3] = 1

        elif seq[i].upper() == "U":
            row[3] = 1

        tensor.append(row)

r1 = re.compile(r"([ACGNTUacgntu]+)\s+([\d+])")
length = []
tensor = [] # Tensor of input sequence information
clabel = [] # List of class labels

with gzip.open(args.data, "rt") as f:
    for line in f:
        m1 = r1.match(line)

        if m1:
            seqlen = len(m1.group(1))
            
            if seqlen <= args.matrix:
                length.append(seqlen)
                
                # Convert a sequence into a matrix (one-hot format)
                seq2mat(tensor, m1.group(1))

                for i in range(args.matrix-seqlen):
                    row = [0, 0, 0, 0]
                    tensor.append(row)
                
                clabel.append(int(m1.group(2)))

            else:
                print("Error: w <= M must hold.")
                print("w: window size")
                print("M: max matrix row size")
                sys.exit()

import numpy as np

count = len(length)
tensor = np.array(tensor)
tensor = tensor.reshape(count, args.matrix, 4)
clabel = np.array(clabel)

import keras

# Set parameters
num_classes = 2
input_shape = (args.matrix, 4)

# Convert class vectors to binary class matrices
clabel = keras.utils.to_categorical(clabel, num_classes)

verbose = 0

if args.train:
    from keras.models import Sequential
    from keras.layers.wrappers import Bidirectional
    from keras.layers.recurrent import LSTM
    from keras.layers import Dense, Dropout
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    # Retrive the output file prefix
    r2 = re.compile(r"(\S+)\.h5$")
    prefix = "model"
    m2 = r2.match(args.model)

    if m2:
        prefix = m2.group(1)
    
    # Construct a model
    model = Sequential()
    model.add(Bidirectional(LSTM(int(args.unit / 2)), input_shape = (args.matrix, 4)))
    model.add(Dropout(args.dropout))
    model.add(Dense(num_classes, activation = 'softmax'))

    if args.verbose:
        # Show the summary of the model
        model.summary()
        verbose = 1

    # Set the training process
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999),
                  metrics = ['accuracy'])

    # Train the model
    # Validation will be done with the last 10% of the data (always fixed)
    #hist = model.fit(tensor, clabel, batch_size = args.batch, epochs = args.epoch,
                     #verbose = verbose, validation_split = 0.1)
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 2, verbose = verbose)
    file_path = "%s.e{epoch:02d}.l{val_loss:.2f}.h5" % prefix
    model_checkpoint = ModelCheckpoint(file_path, monitor = 'val_loss', verbose = verbose,
                                       save_best_only = True, save_weights_only = False, mode = 'auto',
                                       period = 1)
    hist = model.fit(tensor, clabel, batch_size = args.batch, epochs = args.epoch,
                     verbose = verbose, validation_split = 0.1,
                     callbacks = [early_stopping, model_checkpoint])

    # Show each accuracy
    print("Accuracy on the training set:")
    print(hist.history['acc'])
    print("Accuracy on the validation set:")
    print(hist.history['val_acc'])

    # Save the trained model
    #model.save(args.model)

    # Clear the memory
    from keras.backend import tensorflow_backend as backend

    backend.clear_session()

else:
    from keras.models import load_model
    
    # Load the trained model  
    model = load_model(args.model)

    if args.verbose:
        # Show the summary of the model
        model.summary()
        verbose = 1
        
        # Calculate a loss value and accuracy
        score = model.evaluate(tensor, clabel, batch_size = args.batch, verbose = verbose)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

    # Generate predicted classes (specify the batch size)
    out = model.predict_classes(tensor, batch_size = args.batch, verbose = verbose)
    print(out)

    # Clear the memory
    from keras.backend import tensorflow_backend as backend

    backend.clear_session()
