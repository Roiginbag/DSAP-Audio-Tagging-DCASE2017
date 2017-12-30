"""
Summary:  DCASE 2017 task 4 Large-scale weakly supervised 
          sound event detection for smart cars. Ranked 1 in DCASE 2017 Challenge.
Author:   Yong Xu, Qiuqiang Kong
Created:  03/04/2017
Modified: 31/10/2017
"""
from __future__ import print_function 
import sys
import cPickle
import numpy as np
import argparse
import glob
import time
import os

import keras
from keras import backend as K
from keras.models import Sequential,Model, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute,Lambda, RepeatVector
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Conv2D,MaxPooling2D, Convolution1D,MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling2D,GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.layers import Merge, Input, merge
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
import h5py
from keras.layers.merge import Multiply
from sklearn import preprocessing
import random

import config_v2 as cfg
from prepare_data import create_folder, load_hdf5_data, do_scale
from data_generator import RatioDataGenerator, BalanceDataGenerator
from evaluation import io_task4, evaluate

# CNN with Gated linear unit (GLU) block
def block(input):
    cnn = Conv2D(128, (3, 3), padding="same", activation="linear", use_bias=False)(input)
    cnn = BatchNormalization(axis=-1)(cnn)

    cnn1 = Lambda(slice1, output_shape=slice1_output_shape)(cnn)
    cnn2 = Lambda(slice2, output_shape=slice2_output_shape)(cnn)

    cnn1 = Activation('linear')(cnn1)
    cnn2 = Activation('sigmoid')(cnn2)

    out = Multiply()([cnn1, cnn2])
    return out

def double_conv_block(input):
    out = BatchNormalization(axis=-1)(input)
    out = Activation('relu')(out)
    out = Conv2D(64, (3,3), padding="same", activation="linear")(out)
    out = BatchNormalization(axis=-1)(out)
    out = Activation('relu')(out)
    out = Conv2D(64, (3,3), padding="same", activation="linear")(out)
    return out

def slice1(x):
    return x[:, :, :, 0:64]

def slice2(x):
    return x[:, :, :, 64:128]

def slice1_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],64])

def slice2_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],64])

# Attention weighted sum
def outfunc(vects):
    cla, att = vects    # (N, n_time, n_out), (N, n_time, n_out)
    att = K.clip(att, 1e-7, 1.)
    out = K.sum(cla * att, axis=1) / K.sum(att, axis=1)     # (N, n_out)
    return out

def gated_model_30_128(n_time, n_freq, num_classes):
    input_logmel = Input(shape=(n_time, n_freq), name='in_layer')   # (N, 30, 128)
    a1 = Reshape((n_time, n_freq, 1))(input_logmel) # (N, 30, 128, 1)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 30, 64, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 30, 32, 128)

    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 30, 16, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 30, 8, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 30, 4, 128)
    
    a1 = Conv2D(128, (3, 3), padding="same", activation="relu", use_bias=True)(a1)
    a1 = MaxPooling2D(pool_size=(1, 4))(a1) # (N, 30, 1, 128)
    
    a1 = Reshape((30, 128))(a1) # (N, 60, 128)
    
    # Gated BGRU
    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(a1)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(a1)
    a2 = Multiply()([rnnout, rnnout_gate])
    
    # Attention
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(a2)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(a2)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])
    
    model = Model(input_logmel, out)

    return model
def gated_model_60_128(n_time, n_freq, num_classes):
    input_logmel = Input(shape=(n_time, n_freq), name='in_layer')   # (N, 60, 128)
    a1 = Reshape((n_time, n_freq, 1))(input_logmel) # (N, 60, 128, 1)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 60, 64, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 60, 32, 128)

    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 60, 16, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 60, 8, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 60, 4, 128)
    
    a1 = Conv2D(128, (3, 3), padding="same", activation="relu", use_bias=True)(a1)
    a1 = MaxPooling2D(pool_size=(1, 4))(a1) # (N, 60, 1, 128)
    
    a1 = Reshape((60, 128))(a1) # (N, 60, 128)
    
    # Gated BGRU
    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(a1)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(a1)
    a2 = Multiply()([rnnout, rnnout_gate])
    
    # Attention
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(a2)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(a2)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])
    
    model = Model(input_logmel, out)

    return model
def gated_model_120_128(n_time, n_freq, num_classes):
    input_logmel = Input(shape=(n_time, n_freq), name='in_layer')   # (N, 120, 128)
    a1 = Reshape((n_time, n_freq, 1))(input_logmel) # (N, 120, 128, 1)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 120, 64, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 120, 32, 128)

    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 120, 16, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 120, 8, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 120, 4, 128)
    
    a1 = Conv2D(256, (3, 3), padding="same", activation="relu", use_bias=True)(a1)
    a1 = MaxPooling2D(pool_size=(1, 4))(a1) # (N, 120, 1, 256)
    
    a1 = Reshape((120, 256))(a1) # (N, 360, 256)
    
    # Gated BGRU
    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(a1)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(a1)
    a2 = Multiply()([rnnout, rnnout_gate])
    
    # Attention
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(a2)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(a2)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])
    
    model = Model(input_logmel, out)

    return model
def gated_model_120_64(n_time, n_freq, num_classes):
    input_logmel = Input(shape=(n_time, n_freq), name='in_layer')   # (N, 120, 64)
    a1 = Reshape((n_time, n_freq, 1))(input_logmel) # (N, 120, 64, 1)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 120, 32, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 120, 16, 128)

    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 120, 8, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 120, 4, 128)
    
    a1 = Conv2D(256, (3, 3), padding="same", activation="relu", use_bias=True)(a1)
    a1 = MaxPooling2D(pool_size=(1, 4))(a1) # (N, 120, 1, 256)
    
    a1 = Reshape((120, 256))(a1) # (N, 120, 256)
    
    # Gated BGRU
    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(a1)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(a1)
    a2 = Multiply()([rnnout, rnnout_gate])
    
    # Attention
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(a2)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(a2)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])
    
    model = Model(input_logmel, out)
def gated_model_240_64(n_time, n_freq, num_classes):
    input_logmel = Input(shape=(n_time, n_freq), name='in_layer')   # (N, 240, 64)
    a1 = Reshape((n_time, n_freq, 1))(input_logmel) # (N, 240, 64, 1)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 32, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 16, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 8, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 4, 128)
    
    a1 = Conv2D(256, (3, 3), padding="same", activation="relu", use_bias=True)(a1)
    a1 = MaxPooling2D(pool_size=(1, 4))(a1) # (N, 240, 1, 256)
    
    a1 = Reshape((240, 256))(a1) # (N, 240, 256)
    
    # Gated BGRU
    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(a1)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(a1)
    a2 = Multiply()([rnnout, rnnout_gate])
    
    # Attention
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(a2)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(a2)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])
    
    model = Model(input_logmel, out)

    return model
def gated_model_240_128(n_time, n_freq, num_classes):
    input_logmel = Input(shape=(n_time, n_freq), name='in_layer')   # (N, 240, 256)
    a1 = Reshape((n_time, n_freq, 1))(input_logmel) # (N, 240, 128, 1)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 64, 128)

    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 32, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 16, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 8, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 4, 128)
    
    a1 = Conv2D(256, (3, 3), padding="same", activation="relu", use_bias=True)(a1)
    a1 = MaxPooling2D(pool_size=(1, 4))(a1) # (N, 240, 1, 256)
    
    a1 = Reshape((240, 256))(a1) # (N, 240, 256)
    
    # Gated BGRU
    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(a1)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(a1)
    a2 = Multiply()([rnnout, rnnout_gate])
    
    # Attention
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(a2)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(a2)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])
    
    model = Model(input_logmel, out)

    return model

def gated_model_120_256(n_time, n_freq, num_classes):
    input_logmel = Input(shape=(n_time, n_freq), name='in_layer')   # (N, 120, 256)
    a1 = Reshape((n_time, n_freq, 1))(input_logmel) # (N, 120, 256, 1)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 120, 128, 128)

    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 120, 64, 128)

    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 120, 32, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 120, 16, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 120, 8, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 120, 4, 128)
    
    a1 = Conv2D(256, (3, 3), padding="same", activation="relu", use_bias=True)(a1)
    a1 = MaxPooling2D(pool_size=(1, 4))(a1) # (N, 120, 1, 256)
    
    a1 = Reshape((120, 256))(a1) # (N, 120, 256)
    
    # Gated BGRU
    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(a1)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(a1)
    a2 = Multiply()([rnnout, rnnout_gate])
    
    # Attention
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(a2)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(a2)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])
    
    model = Model(input_logmel, out)

    return model
def gated_model_240_256(n_time, n_freq, num_classes):
    input_logmel = Input(shape=(n_time, n_freq), name='in_layer')   # (N, 240, 256)
    a1 = Reshape((n_time, n_freq, 1))(input_logmel) # (N, 240, 256, 1)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 128, 128)

    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 64, 128)

    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 32, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 16, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 8, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 4, 128)
    
    a1 = Conv2D(256, (3, 3), padding="same", activation="relu", use_bias=True)(a1)
    a1 = MaxPooling2D(pool_size=(1, 4))(a1) # (N, 240, 1, 256)
    
    a1 = Reshape((240, 256))(a1) # (N, 240, 256)
    
    # Gated BGRU
    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(a1)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(a1)
    a2 = Multiply()([rnnout, rnnout_gate])
    
    # Attention
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(a2)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(a2)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])
    
    model = Model(input_logmel, out)

    return model

def gated_model_360_64(n_time, n_freq, num_classes):
    input_logmel = Input(shape=(n_time, n_freq), name='in_layer')   # (N, 360, 64)
    a1 = Reshape((n_time, n_freq, 1))(input_logmel) # (N, 360, 64, 1)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 360, 32, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 360, 16, 128)

    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 360, 8, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 360, 4, 128)
    
    a1 = Conv2D(256, (3, 3), padding="same", activation="relu", use_bias=True)(a1)
    a1 = MaxPooling2D(pool_size=(1, 4))(a1) # (N, 360, 1, 256)
    
    a1 = Reshape((360, 256))(a1) # (N, 360, 256)
    
    # Gated BGRU
    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(a1)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(a1)
    a2 = Multiply()([rnnout, rnnout_gate])
    
    # Attention
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(a2)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(a2)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])
    
    model = Model(input_logmel, out)

    return model
def gated_model_360_128(n_time, n_freq, num_classes):
    input_logmel = Input(shape=(n_time, n_freq), name='in_layer')   # (N, 360, 128)
    a1 = Reshape((n_time, n_freq, 1))(input_logmel) # (N, 360, 128, 1)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 360, 64, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 360, 32, 128)

    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 360, 16, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 360, 8, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 360, 4, 128)
    
    a1 = Conv2D(256, (3, 3), padding="same", activation="relu", use_bias=True)(a1)
    a1 = MaxPooling2D(pool_size=(1, 4))(a1) # (N, 180, 1, 256)
    
    a1 = Reshape((360, 256))(a1) # (N, 360, 256)
    
    # Gated BGRU
    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(a1)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(a1)
    a2 = Multiply()([rnnout, rnnout_gate])
    
    # Attention
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(a2)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(a2)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])
    
    model = Model(input_logmel, out)

    return model
def gated_model_480_128(n_time, n_freq, num_classes):
    input_logmel = Input(shape=(n_time, n_freq), name='in_layer')   # (N, 480, 128)
    a1 = Reshape((n_time, n_freq, 1))(input_logmel) # (N, 480, 128, 1)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 480, 64, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 480, 32, 128)

    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 480, 16, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 480, 8, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 480, 4, 128)
    
    a1 = Conv2D(256, (3, 3), padding="same", activation="relu", use_bias=True)(a1)
    a1 = MaxPooling2D(pool_size=(1, 4))(a1) # (N, 480, 1, 256)
    
    a1 = Reshape((480, 256))(a1) # (N, 480, 256)
    
    # Gated BGRU
    rnnout = Bidirectional(GRU(256, activation='linear', return_sequences=True))(a1)
    rnnout_gate = Bidirectional(GRU(256, activation='sigmoid', return_sequences=True))(a1)
    a2 = Multiply()([rnnout, rnnout_gate])
    
    # Attention
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(a2)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(a2)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])
    
    model = Model(input_logmel, out)

    return model

def gated_model(n_time, n_freq, num_classes):
    input_logmel = Input(shape=(n_time, n_freq), name='in_layer')   # (N, 240, 64)
    a1 = Reshape((n_time, n_freq, 1))(input_logmel) # (N, 240, 64, 1)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 32, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 16, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 8, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 4, 128)
    
    a1 = Conv2D(256, (3, 3), padding="same", activation="relu", use_bias=True)(a1)
    a1 = MaxPooling2D(pool_size=(1, 4))(a1) # (N, 240, 1, 256)
    a1 = Dropout(0.5)(a1)
    
    a1 = Reshape((240, 256))(a1) # (N, 240, 256)
    
    # Gated BGRU
    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(a1)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(a1)
    a2 = Multiply()([rnnout, rnnout_gate])
    
    # Attention
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(a2)
    cla = Dropout(0.5)(cla)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(a2)
    att = Dropout(0.5)(att)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])
    
    model = Model(input_logmel, out)
    model.summary()

    return model
def global_iput_stream_240_128(n_time, n_freq, num_classes):
    input_logmel = Input(shape=(n_time, n_freq), name='in_layer')   # (N, 240, 128)
    a1 = Reshape((n_time, n_freq, 1))(input_logmel) # (N, 240, 128, 1)
    
    a1 = double_conv_block(a1)
    a1 = MaxPooling2D(pool_size=(5, 2))(a1) # (N, 48, 64, 64)

    a1 = double_conv_block(a1)
    a1 = MaxPooling2D(pool_size=(2, 4))(a1) # (N, 24, 16, 64)

    a1 = double_conv_block(a1)
    a1 = MaxPooling2D(pool_size=(3, 2))(a1) # (N, 8, 8, 64)

    a1 = double_conv_block(a1)
    a1 = MaxPooling2D(pool_size=(2, 2))(a1) # (N, 4, 4, 64)

    a1 = Reshape((1, 1024))(a1)
    a1 = GlobalAveragePooling1D()(a1)
    out = Dense(17, activation='softmax')(a1)

    model = Model(input_logmel, out)
    
    return model
def global_iput_stream_1024_128(n_time, n_freq, num_classes):
    input_logmel = Input(shape=(n_time, n_freq), name='in_layer')   # (N, 240, 128)
    a1 = Reshape((n_time, n_freq, 1))(input_logmel) # (N, 1024, 128, 1)
    
    a1 = double_conv_block(a1)
    a1 = MaxPooling2D(pool_size=(4, 4))(a1) # (N, 256, 32, 64)

    a1 = double_conv_block(a1)
    a1 = MaxPooling2D(pool_size=(4, 4))(a1) # (N, 64, 8, 64)

    a1 = double_conv_block(a1)
    a1 = MaxPooling2D(pool_size=(4, 2))(a1) # (N, 16, 4, 64)

    a1 = double_conv_block(a1)
    a1 = MaxPooling2D(pool_size=(4, 1))(a1) # (N, 4, 4, 64)

    a1 = Reshape((1, 1024))(a1)
    a1 = GlobalAveragePooling1D()(a1)
    out = Dense(17, activation='softmax', name='localization_layer')(a1)

    model = Model(input_logmel, out)
    
    return model

# Train model
def train(args):
    num_classes = cfg.num_classes
    
    # Load training & testing data
    (tr_x, tr_y, tr_na_list) = load_hdf5_data(args.tr_hdf5_path, verbose=1)
    #(te_x, te_y, te_na_list) = load_hdf5_data(args.te_hdf5_path, verbose=1)
    print("tr_x.shape: %s" % (tr_x.shape,))

    # Scale data
    tr_x = do_scale(tr_x, args.scaler_path, verbose=1)
    te_x = tr_x[45000:]
    te_y = tr_y[45000:]
    tr_x = tr_x[:45000]
    tr_y = tr_y[:45000]
    #te_x = do_scale(te_x, args.scaler_path, verbose=1)
    
    # Build model
    (_, n_time, n_freq) = tr_x.shape    # (N, 240, 64)
    #model = gated_model(n_time, n_freq, num_classes)
    model = gated_model_240_128(n_time, n_freq, num_classes)
    model.summary()
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    # Save model callback
    filepath = os.path.join(args.out_model_dir, "extractor_240_128_.{epoch:02d}-{val_acc:.4f}.hdf5")
    create_folder(os.path.dirname(filepath))
    save_model = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc', 
                                 verbose=0,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1)  

    # Data generator
    gen = RatioDataGenerator(batch_size=24, type='train')

    # Train
    model.fit_generator(generator=gen.generate([tr_x], [tr_y]), 
                        steps_per_epoch=100,    # 100 iters is called an 'epoch'
                        epochs=40,              # Maximum 'epoch' to train
                        verbose=1, 
                        callbacks=[save_model], 
                        validation_data=(te_x, te_y))

# Run function in mini-batch to save memory. 
def run_func(func, x, batch_size):
    pred_all = []
    batch_num = int(np.ceil(len(x) / float(batch_size)))
    for i1 in xrange(batch_num):
        batch_x = x[batch_size * i1 : batch_size * (i1 + 1)]
        [preds] = func([batch_x, 0.])
        pred_all.append(preds)
    pred_all = np.concatenate(pred_all, axis=0)
    return pred_all

def gated_original_model_for_extraction(n_time, n_freq, num_classes, model_path):
    input_logmel = Input(shape=(n_time, n_freq), name='in_layer')   # (N, 240, 64)
    a1 = Reshape((n_time, n_freq, 1))(input_logmel) # (N, 240, 64, 1)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 32, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 16, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 8, 128)
    
    a1 = block(a1)
    a1 = block(a1)
    a1 = MaxPooling2D(pool_size=(1, 2))(a1) # (N, 240, 4, 128)
    
    a1 = Conv2D(256, (3, 3), padding="same", activation="relu", use_bias=True)(a1)
    a1 = MaxPooling2D(pool_size=(1, 4))(a1) # (N, 240, 1, 256)
    
    a1 = Reshape((240, 256))(a1) # (N, 240, 256)
    
    # Gated BGRU
    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(a1)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(a1)
    a2 = Multiply()([rnnout, rnnout_gate])
    
    # Attention
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(a2)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(a2)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])
    
    model = Model(input_logmel, out)
    [model_path] = glob.glob(os.path.join(args.model_dir, "*.%02d-0.*.hdf5"))
    model = load_model(model_weights_path)

    model = Model(input_logmel, a1)
    return model

def ensamble_model(n_t, n_f, num_classes):
    input_logmel = Input(shape=(n_t, n_f), name='in_layer')   # (N, 480, 128)
    a1 = Reshape((n_t,n_f, 1))(input_logmel) # (N, 480, 128, 1)

    a1 = Conv2D(8, (3,3), padding="same", activation="relu")(a1)
    a1 = Conv2D(8, (3,3), padding="same", activation="relu")(a1)
    a1 = MaxPooling2D(pool_size=(2, 2))(a1) # (N, 480, 16, 128)

    a1 = Conv2D(8, (3,3), padding="same", activation="relu")(a1)
    a1 = Conv2D(8, (3,3), padding="same", activation="relu")(a1)
    a1 = MaxPooling2D(pool_size=(2, 2))(a1) # (N, 480, 16, 128)

    a1 = Conv2D(16, (3,3), padding="same", activation="relu")(a1)
    a1 = Conv2D(16, (3,3), padding="same", activation="relu")(a1)
    a1 = MaxPooling2D(pool_size=(2, 2))(a1) # (N, 480, 16, 128)

    a1 = Conv2D(16, (3,3), padding="same", activation="relu")(a1)
    a1 = Conv2D(16, (3,3), padding="same", activation="relu")(a1)
    a1 = MaxPooling2D(pool_size=(2, 2))(a1) # (N, 480, 16, 128)

    a1 = Conv2D(32, (3,3), padding="same",activation="relu")(a1)
    a1 = Conv2D(32, (3,3),  padding="same",activation="relu")(a1)
    a1 = MaxPooling2D(pool_size=(2, 2))(a1) # (N, 480, 16, 128)

    a1 = Conv2D(32, (3,3), padding="same",activation="relu")(a1)
    a1 = Conv2D(32, (3,3),  padding="same",activation="relu")(a1)
    a1 = MaxPooling2D(pool_size=(2, 2))(a1) # (N, 480, 16, 128)

    a1 = Flatten()(a1)
    a1 = Dense(1024)(a1)
    a1 = Dropout(0.75)(a1)
    a1 = Dense(1024)(a1)
    a1 = Dropout(0.75)(a1)
    #a1 = Dense(1024)(a1)
    out = Dense(num_classes, activation='softmax')(a1)

    model = Model(input_logmel, out)

    return model

def ensamble_model_2(n_t, n_f, num_classes):
    input_logmel = Input(shape=(n_t, n_f), name='in_layer')   # (N, 480, 128)
    # Gated BGRU
    rnnout = Bidirectional(GRU(128, activation='linear', return_sequences=True))(input_logmel)
    rnnout_gate = Bidirectional(GRU(128, activation='sigmoid', return_sequences=True))(input_logmel)
    a2 = Multiply()([rnnout, rnnout_gate])
    
    # Attention
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(a2)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(a2)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])
    
    model = Model(input_logmel, out)

    return model

def extract_features_for_ensamble(args):
    num_classes = cfg.num_classes
    t1 = time.time()
    # Load training & testing data
    (tr_x, tr_y, tr_na_list) = load_hdf5_data(args.tr_hdf5_path, verbose=1)
    (te_x, te_y, te_na_list) = load_hdf5_data(args.te_hdf5_path, verbose=1)
    print("tr_x.shape: %s" % (tr_x.shape,))
    pred_at_list = []
    # Scale data
    tr_x = do_scale(tr_x, args.scaler_path, verbose=1)
    te_x = do_scale(te_x, args.scaler_path, verbose=1)
    
    # Build model
    (_, n_time, n_freq) = tr_x.shape    # (N, 240, 64)

    # Create feautre extractor model
    model = load_model(args.model_path)
    in_layer = model.get_layer('in_layer')
    resh_layer = model.get_layer('reshape_2')
    model = Model(in_layer.input, resh_layer.output)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train
    # Batch feature extraction
    batch_size = 20
    batch_num = int(np.ceil(len(tr_x) / float(batch_size)))
    for i1 in xrange(batch_num):
        batch_x = tr_x[batch_size * i1 : batch_size * (i1 + 1)]
        preds = model.predict(batch_x)
        pred_at_list.append(preds)
    pred_at_list = np.concatenate(pred_at_list, axis=0)
    #pred = model.predict(tr_x)
    #pred_at_list.append(pred)
    pred_at_list = np.array(pred_at_list, dtype=np.float32)
    print("Pred at list shape:", pred_at_list.shape)
    print("Prediction time: %s" % (time.time() - t1,))

    create_folder(args.out_dir)
    tr_y = np.array(tr_y, dtype=np.bool)
    out_path = args.out_dir+"/ensamble_features_train_"+str(n_time)+"_"+str(n_freq)+".hdf5"
    hf = h5py.File(out_path, 'w')
    hf.create_dataset('x', data=pred_at_list)
    hf.create_dataset('y', data=tr_y)
    hf.close()
    # Test
    pred_at_list = []
    batch_num = int(np.ceil(len(te_x) / float(batch_size)))
    for i1 in xrange(batch_num):
        batch_x = te_x[batch_size * i1 : batch_size * (i1 + 1)]
        preds = model.predict(batch_x)
        pred_at_list.append(preds)
    pred_at_list = np.concatenate(pred_at_list, axis=0)
    #pred = model.predict(tr_x)
    #pred_at_list.append(pred)
    pred_at_list = np.array(pred_at_list, dtype=np.float32)
    print("Pred at list shape:", pred_at_list.shape)
    print("Prediction time: %s" % (time.time() - t1,))

    te_y = np.array(te_y, dtype=np.bool)
    out_path = args.out_dir+"/ensamble_features_test_"+str(n_time)+"_"+str(n_freq)+".hdf5"
    hf = h5py.File(out_path, 'w')
    hf.create_dataset('x', data=pred_at_list)
    hf.create_dataset('y', data=te_y)
    hf.close()

def train_classifier_2inputs(args):
    num_classes = cfg.num_classes
    
    # Load training & testing data
    hf = h5py.File(args.tr_hdf5_path, 'r')
    tr_x = np.array(hf.get('x'))
    tr_y = np.array(hf.get('y'))
    hf.close()
    
    print("tr_x.shape: %s" % (tr_x.shape,))
    print("tr_y.shape: %s" % (tr_y.shape,))

    #tr_y = to_categorical(tr_y, num_classes)
    #te_y = to_categorical(te_y, num_classes)
    
    hf = h5py.File(args.tr_hdf5_path_2, 'r')
    tr_x_2 = np.array(hf.get('x'))
    tr_y_2 = np.array(hf.get('y'))
    hf.close()
    
    #(te_x, te_y, te_na_list) = load_hdf5_data(args.te_hdf5_path, verbose=1)
    print("tr_x_2.shape: %s" % (tr_x_2.shape,))
    print("tr_y_2.shape: %s" % (tr_y_2.shape,))
    
    tr_x_all = np.concatenate((tr_x, tr_x_2), axis=1)
    print("tr_x_all.shape: %s" % (tr_x_all.shape,))
    # Build model
    (_, n_time, n_freq) = tr_x_all.shape    # (N, 240, 64)
    model = ensamble_model_2(n_time, n_freq, num_classes)

    model.summary()
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Save model callback
    filepath = os.path.join(args.out_model_dir, "classif_.{epoch:02d}-{val_acc:.4f}.hdf5")
    create_folder(os.path.dirname(filepath))
    save_model = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc', 
                                 verbose=0,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1)  

    # Data generator
    gen = RatioDataGenerator(batch_size=24, type='train')

    # Train
    model.fit_generator(generator=gen.generate([tr_x_all[:45000]], [tr_y[:45000]]), 
                        steps_per_epoch=100,    # 100 iters is called an 'epoch'
                        epochs=300,              # Maximum 'epoch' to train
                        verbose=2, 
                        callbacks=[save_model], 
                        validation_data=(tr_x_all[45000:], tr_y[45000:]))

def train_classifier_3inputs(args):
    num_classes = cfg.num_classes
    
    # Load training & testing data
    hf = h5py.File(args.tr_hdf5_path, 'r')
    tr_x = np.array(hf.get('x'))
    tr_y = np.array(hf.get('y'))
    hf.close()
    
    print("tr_x.shape: %s" % (tr_x.shape,))
    print("tr_y.shape: %s" % (tr_y.shape,))

    #tr_y = to_categorical(tr_y, num_classes)
    #te_y = to_categorical(te_y, num_classes)
    
    hf = h5py.File(args.tr_hdf5_path_2, 'r')
    tr_x_2 = np.array(hf.get('x'))
    tr_y_2 = np.array(hf.get('y'))
    hf.close()
    
    #(te_x, te_y, te_na_list) = load_hdf5_data(args.te_hdf5_path, verbose=1)
    print("tr_x_2.shape: %s" % (tr_x_2.shape,))
    print("tr_y_2.shape: %s" % (tr_y_2.shape,))

    hf = h5py.File(args.tr_hdf5_path_3, 'r')
    tr_x_3 = np.array(hf.get('x'))
    tr_y_3 = np.array(hf.get('y'))
    hf.close()
    
    #(te_x, te_y, te_na_list) = load_hdf5_data(args.te_hdf5_path, verbose=1)
    print("tr_x_3.shape: %s" % (tr_x_3.shape,))
    print("tr_y_3.shape: %s" % (tr_y_3.shape,))
    
    tr_x_all = np.concatenate((tr_x, tr_x_2, tr_x_3), axis=1)
    print("tr_x_all.shape: %s" % (tr_x_all.shape,))
    # Build model
    (_, n_time, n_freq) = tr_x_all.shape    # (N, 240, 64)
    model = ensamble_model(n_time, n_freq, num_classes)

    model.summary()
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Save model callback
    filepath = os.path.join(args.out_model_dir, "classif_.{epoch:02d}-{val_acc:.4f}.hdf5")
    create_folder(os.path.dirname(filepath))
    save_model = ModelCheckpoint(filepath=filepath,
                                 monitor='val_acc', 
                                 verbose=0,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1)  

    # Data generator
    gen = RatioDataGenerator(batch_size=24, type='train')

    # Train
    model.fit_generator(generator=gen.generate([tr_x_all[:45000]], [tr_y[:45000]]), 
                        steps_per_epoch=100,    # 100 iters is called an 'epoch'
                        epochs=300,              # Maximum 'epoch' to train
                        verbose=2, 
                        callbacks=[save_model], 
                        validation_data=(tr_x_all[45000:], tr_y[45000:]))

# Recognize and write probabilites. 
def recognize(args, at_bool, sed_bool):
    (te_x, te_y, te_na_list) = load_hdf5_data(args.te_hdf5_path, verbose=1)
    x = te_x
    y = te_y
    na_list = te_na_list
    
    x = do_scale(x, args.scaler_path, verbose=1)
    
    fusion_at_list = []
    fusion_sed_list = []
    for epoch in range(30, 40, 1):
        t1 = time.time()
        [model_path] = glob.glob(os.path.join(args.model_dir, 
            "*.%02d-0.*.hdf5" % epoch))
        model = load_model(model_path)
        
        # Audio tagging
        if at_bool:
            pred = model.predict(x)
            fusion_at_list.append(pred)
        
        # Sound event detection
        if sed_bool:
            in_layer = model.get_layer('in_layer')
            loc_layer = model.get_layer('localization_layer')
            func = K.function([in_layer.input, K.learning_phase()], 
                              [loc_layer.output])
            pred3d = run_func(func, x, batch_size=20)
            fusion_sed_list.append(pred3d)
        
        print("Prediction time: %s" % (time.time() - t1,))
    
    # Write out AT probabilities
    if at_bool:
        fusion_at = np.mean(np.array(fusion_at_list), axis=0)
        print("AT shape: %s" % (fusion_at.shape,))
        io_task4.at_write_prob_mat_to_csv(
            na_list=na_list, 
            prob_mat=fusion_at, 
            out_path=os.path.join(args.out_dir, "at_prob_mat.csv.gz"))
    
    # Write out SED probabilites
    if sed_bool:
        fusion_sed = np.mean(np.array(fusion_sed_list), axis=0)
        print("SED shape:%s" % (fusion_sed.shape,))
        io_task4.sed_write_prob_mat_list_to_csv(
            na_list=na_list, 
            prob_mat_list=fusion_sed, 
            out_path=os.path.join(args.out_dir, "sed_prob_mat_list.csv.gz"))
            
    print("Prediction finished!")

# Recognize and write probabilites. 
def recognize_trained(args, at_bool, sed_bool):
    (te_x, te_y, te_na_list) = load_hdf5_data(args.te_hdf5_path, verbose=1)
    x = te_x
    y = te_y
    na_list = te_na_list
    
    x = do_scale(x, args.scaler_path, verbose=1)
    
    fusion_at_list = []
    fusion_sed_list = []
    t1 = time.time()
    [model_path] = glob.glob(os.path.join(args.model_dir, "*.35-0.*.hdf5"))
    model = load_model(model_path)
    
    # Audio tagging
    if at_bool:
        pred = model.predict(x)
        fusion_at_list.append(pred)
    
    # Sound event detection
    if sed_bool:
        in_layer = model.get_layer('in_layer')
        loc_layer = model.get_layer('localization_layer')
        func = K.function([in_layer.input, K.learning_phase()], 
                          [loc_layer.output])
        pred3d = run_func(func, x, batch_size=20)
        fusion_sed_list.append(pred3d)
    
    print("Prediction time: %s" % (time.time() - t1,))
    
    # Write out AT probabilities
    if at_bool:
        fusion_at = np.mean(np.array(fusion_at_list), axis=0)
        print("AT shape: %s" % (fusion_at.shape,))
        io_task4.at_write_prob_mat_to_csv(
            na_list=na_list, 
            prob_mat=fusion_at, 
            out_path=os.path.join(args.out_dir, "at_prob_mat.csv.gz"))
    
    # Write out SED probabilites
    if sed_bool:
        fusion_sed = np.mean(np.array(fusion_sed_list), axis=0)
        print("SED shape:%s" % (fusion_sed.shape,))
        io_task4.sed_write_prob_mat_list_to_csv(
            na_list=na_list, 
            prob_mat_list=fusion_sed, 
            out_path=os.path.join(args.out_dir, "sed_prob_mat_list.csv.gz"))
            
    print("Prediction finished!")

# Recognize and write probabilites. 
def recognize_trained_classif(args):
    (_, _, te_na_list) = load_hdf5_data(args.feat_te_hdf5_path, verbose=1)
    hf = h5py.File(args.te_hdf5_path, 'r')
    te_x = np.array(hf.get('x'))
    te_y = np.array(hf.get('y'))
    hf.close()
    hf = h5py.File(args.te_hdf5_path_2, 'r')
    te_x_2 = np.array(hf.get('x'))
    te_y_2 = np.array(hf.get('y'))
    hf.close()
    """
    hf = h5py.File(args.te_hdf5_path_3, 'r')
    te_x_3 = np.array(hf.get('x'))
    te_y_3 = np.array(hf.get('y'))
    hf.close()
    """
    te_x_all = np.concatenate((te_x, te_x_2), axis=1)

    t1 = time.time()
    model = load_model(args.model_path)

    fusion_at = model.predict(te_x_all)

    print("AT shape: %s" % (fusion_at.shape,))
    io_task4.at_write_prob_mat_to_csv(
        na_list=te_na_list, 
        prob_mat=fusion_at, 
        out_path=os.path.join(args.out_dir, "at_prob_mat.csv.gz"))

    print("Prediction finished!")

# Get stats from probabilites. 
def get_stat(args, at_bool, sed_bool):
    lbs = cfg.lbs
    step_time_in_sec = cfg.step_time_in_sec
    max_len = cfg.max_len
    thres_ary = [0.3] * len(lbs)

    # Calculate AT stat
    if at_bool:
        pd_prob_mat_csv_path = os.path.join(args.pred_dir, "at_prob_mat.csv.gz")
        at_stat_path = os.path.join(args.stat_dir, "at_stat.csv")
        at_submission_path = os.path.join(args.submission_dir, "at_submission.csv")
        
        at_evaluator = evaluate.AudioTaggingEvaluate(
            weak_gt_csv="meta_data/groundtruth_weak_label_testing_set.csv", 
            lbs=lbs)
        
        at_stat = at_evaluator.get_stats_from_prob_mat_csv(
                        pd_prob_mat_csv=pd_prob_mat_csv_path, 
                        thres_ary=thres_ary)
                        
        # Write out & print AT stat
        at_evaluator.write_stat_to_csv(stat=at_stat, 
                                       stat_path=at_stat_path)
        at_evaluator.print_stat(stat_path=at_stat_path)
        
        # Write AT to submission format
        io_task4.at_write_prob_mat_csv_to_submission_csv(
            at_prob_mat_path=pd_prob_mat_csv_path, 
            lbs=lbs, 
            thres_ary=at_stat['thres_ary'], 
            out_path=at_submission_path)
               
    # Calculate SED stat
    if sed_bool:
        sed_prob_mat_list_path = os.path.join(args.pred_dir, "sed_prob_mat_list.csv.gz")
        sed_stat_path = os.path.join(args.stat_dir, "sed_stat.csv")
        sed_submission_path = os.path.join(args.submission_dir, "sed_submission.csv")
        
        sed_evaluator = evaluate.SoundEventDetectionEvaluate(
            strong_gt_csv="meta_data/groundtruth_strong_label_testing_set.csv", 
            lbs=lbs, 
            step_sec=step_time_in_sec, 
            max_len=max_len)
                            
        # Write out & print SED stat
        sed_stat = sed_evaluator.get_stats_from_prob_mat_list_csv(
                    pd_prob_mat_list_csv=sed_prob_mat_list_path, 
                    thres_ary=thres_ary)
                    
        # Write SED to submission format
        sed_evaluator.write_stat_to_csv(stat=sed_stat, 
                                        stat_path=sed_stat_path)                     
        sed_evaluator.print_stat(stat_path=sed_stat_path)
        
        # Write SED to submission format
        io_task4.sed_write_prob_mat_list_csv_to_submission_csv(
            sed_prob_mat_list_path=sed_prob_mat_list_path, 
            lbs=lbs, 
            thres_ary=thres_ary, 
            step_sec=step_time_in_sec, 
            out_path=sed_submission_path)
                                                        
    print("Calculating stat finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest='mode')
    
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--tr_hdf5_path', type=str)
    parser_train.add_argument('--te_hdf5_path', type=str)
    parser_train.add_argument('--scaler_path', type=str)
    parser_train.add_argument('--out_model_dir', type=str)

    parser_train_classif = subparsers.add_parser('train_classif')
    parser_train_classif.add_argument('--tr_hdf5_path', type=str)
    parser_train_classif.add_argument('--tr_hdf5_path_2', type=str)
    #parser_train_classif.add_argument('--tr_hdf5_path_3', type=str)
    parser_train_classif.add_argument('--te_hdf5_path', type=str)
    parser_train_classif.add_argument('--te_hdf5_path_2', type=str)
    #parser_train_classif.add_argument('--te_hdf5_path_3', type=str)
    parser_train_classif.add_argument('--out_model_dir', type=str)
    
    parser_recognize = subparsers.add_parser('recognize')
    parser_recognize.add_argument('--te_hdf5_path', type=str)
    parser_recognize.add_argument('--scaler_path', type=str)
    parser_recognize.add_argument('--model_dir', type=str)
    parser_recognize.add_argument('--out_dir', type=str)

    parser_extractor = subparsers.add_parser('extract_features')
    parser_extractor.add_argument('--tr_hdf5_path', type=str)
    parser_extractor.add_argument('--te_hdf5_path', type=str)
    parser_extractor.add_argument('--scaler_path', type=str)
    parser_extractor.add_argument('--model_path', type=str)
    parser_extractor.add_argument('--out_dir', type=str)

    parser_recognize_trained = subparsers.add_parser('recognize_trained_classif')
    parser_recognize_trained.add_argument('--feat_te_hdf5_path', type=str)
    parser_recognize_trained.add_argument('--te_hdf5_path', type=str)
    parser_recognize_trained.add_argument('--te_hdf5_path_2', type=str)
    #parser_recognize_trained.add_argument('--te_hdf5_path_3', type=str)
    parser_recognize_trained.add_argument('--model_path', type=str)
    parser_recognize_trained.add_argument('--out_dir', type=str)
    
    parser_recognize_trained = subparsers.add_parser('recognize_trained')
    parser_recognize_trained.add_argument('--te_hdf5_path', type=str)
    parser_recognize_trained.add_argument('--scaler_path', type=str)
    parser_recognize_trained.add_argument('--model_dir', type=str)
    parser_recognize_trained.add_argument('--out_dir', type=str)

    parser_get_stat = subparsers.add_parser('get_stat')
    parser_get_stat.add_argument('--pred_dir', type=str)
    parser_get_stat.add_argument('--stat_dir', type=str)
    parser_get_stat.add_argument('--submission_dir', type=str)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'train_classif':
        train_classifier_2inputs(args)
    elif args.mode == 'extract_features':
        extract_features_for_ensamble(args)
    elif args.mode == 'recognize':
        recognize(args, at_bool=True, sed_bool=True)
    elif args.mode == 'recognize_trained':
        recognize_trained(args, at_bool=True, sed_bool=False)
    elif args.mode == 'recognize_trained_classif':
        recognize_trained_classif(args)
    elif args.mode == 'get_stat':
        get_stat(args, at_bool=True, sed_bool=False)
    else:
        raise Exception("Incorrect argument!")