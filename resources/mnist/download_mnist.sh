#!/bin/bash

# If not already downloaded
if [ ! -f ./train-images.idx3-ubyte ]; then
    # If the archive does not exist, download it
    if [ ! -f ./train-images.idx3-ubyte.gz ]; then
        wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
        wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
        wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
        wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    fi

    # Extract all the files
    gunzip *.gz
fi
