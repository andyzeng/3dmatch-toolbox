#!/bin/bash

export PATH=$PATH:/usr/local/cuda/bin

if uname | grep -q Darwin; then
  CUDA_LIB_DIR=/usr/local/cuda/lib
  CUDNN_LIB_DIR=/usr/local/cudnn/v5.1/lib
elif uname | grep -q Linux; then
  CUDA_LIB_DIR=/usr/local/cuda/lib64
  CUDNN_LIB_DIR=/usr/local/cudnn/v5.1/lib64
fi

CUDNN_INC_DIR=/usr/local/cudnn/v5.1/include


# if use opencv, add this into the command line
# `pkg-config --cflags --libs opencv`

nvcc -std=c++11 -O3 -o get3DMatchDescriptors get3DMatchDescriptors.cu -I/usr/local/cuda/include -I$CUDNN_INC_DIR -L$CUDA_LIB_DIR -L$CUDNN_LIB_DIR -lcudart -lcublas -lcudnn -lcurand -D_MWAITXINTRIN_H_INCLUDED `pkg-config --cflags --libs opencv`









