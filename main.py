# -*- coding: utf-8 -*-
"""
Created on Wed Apri 7 16:41:28 2021
@author: guemesturb
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"]=''
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
availale_GPUs = len(physical_devices) 

print('Using TensorFlow version: ', tf.__version__, ', GPU:', availale_GPUs)
print('Using Keras version: ', tf.keras.__version__)

if physical_devices:

  try:

    for gpu in physical_devices:

      tf.config.experimental.set_memory_growth(gpu, True)

  except RuntimeError as e:

    print(e)


import time
from models import model_cnn_mlp
from pipelines import generate_default_training_pipeline
from training import training_loop


def main():

    """
        Define training pipelines
    """

    dataset_train, dataset_valid = generate_default_training_pipeline(tfr_path, channels, n_modes, validation_split, batch_size, shuffle_buffer, n_prefetch, cpu=True)

    """
        Define model
    """

    model = model_cnn_mlp(channels, nz, nx, n_modes, cpu=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model_loss = tf.keras.losses.MeanSquaredError()

    """
        Training loop
    """

    training_loop(dataset_train, dataset_valid, save_path, model_name, model, optimizer, model_loss, epochs)

    return


if __name__ == "__main__":

    nz = 64 
    nx = 128 
    epochs = 100
    yp_flow = 15
    n_modes = 10
    channels = 2 
    n_prefetch = 4
    batch_size = 50
    save_path = ""
    model_name = "test"
    tfr_path = "./data/"
    shuffle_buffer = 5000
    validation_split = 0.2

    main()