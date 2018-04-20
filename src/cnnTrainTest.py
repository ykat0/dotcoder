#!/usr/bin/env python

# cnnTrainTest.py ver. 1.0.0
# Copyright (C) 2017 Yuki Kato
# This script is used to train/test a convolutional neural network (CNN) for predicting if given transcript sequences can form RNA G-quadruplexes (rG4s).
# A trained model will be written to a specified file (e.g. model.h5).
# Note that there is a limit on the matrix row size used for both training and test, which can be specified as an option.
# This script needs "keras," "tensorflow" and "h5py" to be installed.
# Usage: Type "./cnnTrainTest.py -h" in your terminal.

import argparse

parser = argparse.ArgumentParser(
    usage = './cnnTrainTest.py [options]* <data> <model>',
    description = 'This script is used to train/test a convolutional neural network (CNN) for predicting if given transcript sequences can form RNA G-quadruplexes (rG4s). A trained model will be written to a specified file (e.g. model.h5). Note that there is a limit on the matrix row size used for both training and test, which can be specified as an option. This script needs "Keras," "tensorflow" and "h5py" to be installed.',
    epilog = 'cnnTrainTest.py ver. 1.0.0')
parser.add_argument('data', metavar = 'data <STRING>', type = str,
                    help = 'path to the training/test data (sequence, label) in gzipped TSV format')
parser.add_argument('model', metavar = 'model <STRING>', type = str,
                    help = 'path to the trained model in HDF5 format')
parser.add_argument('-v', '--verbose', action = "store_true",
                    help = 'show details (default: off)')
parser.add_argument('-t', '--train', action = "store_true",
                    help = 'training mode (default: off)')
parser.add_argument('-m', '--motif', metavar = '<INT>', type = int,
                    help = 'motif detector length (default <INT> = 20)',
                    default = 20)
parser.add_argument('-f', '--filter', metavar = '<INT>', type = int,
                    help = 'number of filters (default <INT> = 512)',
                    default = 512)
parser.add_argument('-d', '--dropout', metavar = '<FLOAT>', type = float,
                    help = 'dropout rate (default <FLOAT> = 0.25)',
                    default = 0.25)
parser.add_argument('-b', '--batch', metavar = '<INT>', type = int,
                    help = 'batch size (default <INT> = 32)',
                    default = 32)
parser.add_argument('-e', '--epoch', metavar = '<INT>', type = int,
                    help = 'number of epochs (default <INT> = 50)',
                    default = 50)
parser.add_argument('-M', '--matrix', metavar = '<INT>', type = int,
                    help = 'max matrix row size (default <INT> = 300)',
                    default = 300)
args = parser.parse_args()

import re
import gzip
import sys

def seq2mat(tensor, seq):
    # Head padding
    for i in range(args.motif-1):
        row = [0.25, 0.25, 0.25, 0.25]
        tensor.append(row)
    
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

    # Tail padding
    for i in range(args.motif-1):
        row = [0.25, 0.25, 0.25, 0.25]
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
            
            if 2*args.motif+seqlen-2 <= args.matrix:
                length.append(seqlen)
                
                # Convert a sequence into a matrix
                seq2mat(tensor, m1.group(1))

                for i in range(args.matrix-2*args.motif-seqlen+2):
                    row = [0, 0, 0, 0]
                    tensor.append(row)
                
                clabel.append(int(m1.group(2)))

            else:
                print("Error: 2m+w-2 <= M must hold.")
                print("m: motif detector length")
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

# Convert 3D into 4D with channel (1)
tensor = tensor.reshape(count, args.matrix, 4, 1)
input_shape = (args.matrix, 4, 1)

# Convert class vectors to binary class matrices
clabel = keras.utils.to_categorical(clabel, num_classes)

verbose = 0

if args.train:
    from keras.models import Sequential
    from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
    
    # Construct a model
    model = Sequential()
    model.add(Conv2D(args.filter, kernel_size = (args.motif, 4), activation = 'relu',
                     input_shape = input_shape))
    model.add(MaxPooling2D(pool_size = (args.matrix-args.motif+1, 1)))
    model.add(Dropout(args.dropout))
    model.add(Flatten())
    model.add(Dense(num_classes, activation = 'softmax'))

    if args.verbose:
        # Show the summary of the model
        model.summary()
        verbose = 1

    # Set the training process
    model.compile(loss = keras.losses.categorical_crossentropy,
                  optimizer = keras.optimizers.Adadelta(), metrics = ['accuracy'])

    # Training the model with the epoch
    # Validation will be done with the last 10% of the data (always fixed)
    hist = model.fit(tensor, clabel, batch_size = args.batch, epochs = args.epoch,
              verbose = verbose, validation_split = 0.1)

    # Show each accuracy
    print("Accuracy on the training set:")
    print(hist.history['acc'])
    print("Accuracy on the validation set:")
    print(hist.history['val_acc'])

    # Save the trained model
    model.save(args.model)

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
        score = model.evaluate(tensor, clabel, verbose = verbose)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

    # Generate predicted classes (specify the batch size)
    out = model.predict_classes(tensor)
    print(out)

    # Clear the memory
    from keras.backend import tensorflow_backend as backend

    backend.clear_session()
