#!/usr/bin/env python

# rnnPredict.py ver. 1.0.1
# Copyright (C) 2018 Yuki Kato
# This script is used to predict if given transcript sequences can form RNA G-quadruplexes (rG4s) with a bidirectional recurrent neural network (BRNN).
# A trained model for the corresponding window size must be specified.
# Note that there is a limit on the matrix row size used for prediction, which can be specified as an option.
# This script needs "keras," "tensorflow" and "h5py" to be installed.
# Usage: Type "./rnnPredict.py -h" in your terminal.

import argparse

parser = argparse.ArgumentParser(
    usage = './rnnPredict.py [options]* <fasta> <model>',
    description = 'This script is used to predict if given transcript sequences can form RNA G-quadruplexes (rG4s) with a bidirectional recurrent neural network (BRNN). A trained model for the corresponding window size must be specified. Note that there is a limit on the matrix row size used for prediction, which can be specified as an option. This script needs "keras," "tensorflow" and "h5py" to be installed.',
    epilog = 'rnnPredict.py ver. 1.0.1')
parser.add_argument('fasta', metavar = 'fasta <STRING>', type = str,
                    help = 'path to the gzipped multi-FASTA file')
parser.add_argument('model', metavar = 'model <STRING>', type = str,
                    help = 'path to the trained model in HDF5 format (e.g. model.h5)')
parser.add_argument('-v', '--verbose', action = "store_true",
                    help = 'show details (default: off)')
parser.add_argument('-z', '--gzip', action = "store_true",
                    help = 'gzipped input (default: off)')
parser.add_argument('-p', '--prob', action = "store_true",
                    help = 'show output probabilities (default: show predicted classes)')
parser.add_argument('-b', '--batch', metavar = '<INT>', type = int,
                    help = 'batch size (default <INT> = 64)',
                    default = 64)
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

def readFile(obj):
    for count, line in enumerate(obj):
        m1 = r1.match(line)
        
        if not m1:
            seqlen = len(line.strip())
            
            if seqlen <= args.matrix:
                length.append(seqlen)
                
                # Convert a sequence into a matrix (one-hot format)
                seq2mat(tensor, line.strip())
                
                for i in range(args.matrix-seqlen):
                    row = [0, 0, 0, 0]
                    tensor.append(row)
                    
            else:
                print("Error: w <= M must hold.")
                print("w: window size")
                print("M: max matrix row size")
                sys.exit()

# Note: multi-FASTA can be accepted if it contains one sequence per line
r1 = re.compile(r">")
length = []
tensor = [] # Tensor of input sequence information

if args.gzip:
    with gzip.open(args.fasta, "rt") as f:
        readFile(f)

else:
    with open(args.fasta, "r") as f:
        readFile(f)

import numpy as np

count = len(length)
tensor = np.array(tensor)
tensor = tensor.reshape(count, args.matrix, 4)

import keras

# Set parameters
verbose = 0

from keras.models import load_model
    
# Load the trained model  
model = load_model(args.model)

if args.verbose:
    # Show the summary of the model
    model.summary()
    verbose = 1

# Generate predicted classes (specify the batch size)
if args.prob:
    prob = model.predict_proba(tensor, batch_size = args.batch, verbose = verbose)
    print(prob)

else:
    classes = model.predict_classes(tensor, batch_size = args.batch, verbose = verbose)
    print(classes)

# Clear the memory
from keras.backend import tensorflow_backend as backend

backend.clear_session()
