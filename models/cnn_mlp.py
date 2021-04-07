# -*- coding: utf-8 -*-
"""
Created on Wed Apri 7 16:41:28 2021
@author: guemesturb
"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def model_cnn_mlp(channels, nz, nx, n_modes):

    inputs = keras.Input(shape=(channels, nz, nx), name='wall-fields')

    model = layers.Conv2D(filters=int(nx / 2), kernel_size=(4, 4), activation=tf.nn.relu, data_format='channels_first',padding = 'same')(inputs)
    model = layers.MaxPooling2D(pool_size=[2, 2],data_format='channels_first',padding = 'same')(model)
    model = layers.Conv2D(filters=int(nx / 2), kernel_size=(4, 4), activation=tf.nn.relu, data_format='channels_first',padding = 'same')(model)
    model = layers.MaxPooling2D(pool_size=[2, 2], data_format='channels_first',padding = 'same')(model)
    model = layers.Conv2D(filters=int(nx / 2), kernel_size=(4, 4), activation=tf.nn.relu, data_format='channels_first',padding = 'same')(model)
    model = layers.MaxPooling2D(pool_size=[2, 2],data_format='channels_first',padding = 'same')(model)   
    model = layers.Conv2D(filters=int(nx / 2), kernel_size=(4, 4), activation=tf.nn.relu, data_format='channels_first', padding = 'same')(model)   
    model = layers.MaxPooling2D(pool_size=[2, 2],data_format='channels_first',padding = 'same')(model)     
    model = layers.Conv2D(filters=int(nx / 2), kernel_size=(4, 4), activation=tf.nn.relu, data_format='channels_first', padding = 'same')(model)       
    model = layers.MaxPooling2D(pool_size=[2, 2],data_format='channels_first',padding = 'same')(model)  
    model = layers.Flatten()(model)
    model = layers.Dense(int(nx / 2), activation=tf.nn.relu)(model)
    model = layers.Dense(n_modes)(model)
    
    model = keras.Model(inputs, model, name='CNN-POD')

    print(model.summary())

    return model