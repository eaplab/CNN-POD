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


def main():

    """
        Define training pipelines
    """

    dataset_train, dataset_valid = generate_default_training_pipeline(tfr_path, channels, n_modes, validation_split, batch_size, shuffle_buffer, n_prefetch)

    """
        Define model
    """

    model = model_cnn_mlp(channels, nz, nx, n_modes)

    """
        Define training
    """

    """
        Training loop
    """

    model_name = f"Ret_flow-reconstruction_yp{yp_flow:03d}"

    start_time = time.time()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
    )

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error'])

    train_loss = tf.metrics.Mean()
    valid_loss = tf.metrics.Mean()

    for epoch in range(1, epochs + 1):

        train_loss.reset_states()

        for (X_target, Y_target) in dataset_train:
  
            pred_loss, mae = model.train_on_batch(X_target, Y_target)
            train_loss.update_state(pred_loss)
            
        for (X_target, Y_target) in dataset_valid:
            valid_pred_loss, valid_mae = model.test_on_batch(X_target, Y_target)
            valid_loss.update_state(valid_pred_loss)
        
        end_time = time.time()

        if epoch > 10:
        
            predictor.optimizer.lr = 0.001 * tf.math.exp(0.1 * (10 - epoch))

        print(f'Epoch {epoch:04d}/{epochs:04d}, loss: {train_loss.result().numpy()}, val_loss: {valid_loss.result().numpy()}, elapsed time from start: {end_time - start_time}')

    predictor_name = models_path + model_name + '_predictor.tf'
    predictor.save(predictor_name)

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
    tfr_path = "./data/"
    shuffle_buffer = 5000
    validation_split = 0.2

    main()