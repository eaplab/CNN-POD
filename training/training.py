# -*- coding: utf-8 -*-
"""
Created on Wed Apri 7 16:41:28 2021
@author: guemesturb
"""


import os
import time
import tensorflow as tf


def training_loop(dataset_train, dataset_valid, save_path, model_name, model, optimizer, model_loss, epochs, saving_freq=2, pretrained=False):

    log_folder = f"./logs/"

    if not os.path.exists(log_folder):

        os.mkdir(log_folder)

    checkpoint_dir = f"{save_path}checkpoints_{model_name}"

    if not os.path.exists(checkpoint_dir):

        os.makedirs(checkpoint_dir)

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

    if pretrained:

        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        with open(f"{log_folder}/log_{model_name}.log",'r') as fd:

            epoch_bias = int(fd.read().splitlines()[-1].split(',')[0])

        epoch_bias = epoch_bias // saving_freq * saving_freq

        drop = epoch_bias % saving_freq

        if drop != 0:

            with open(f"{log_folder}/log_{model_name}.log", 'r') as fd:
            
                lines = fd.readlines()[:-drop]

            with open(f"{log_folder}/log_{model_name}.log", 'w') as fd:

                fd.writelines(lines)

        print(f"Training reinitialized after {epoch_bias} epochs")

    else:

        with open(f"{log_folder}/log_{model_name}.log",'w') as fd:

            fd.write(f"epoch,loss,val_loss,time\n")

        epoch_bias = 0

    train_loss = tf.metrics.Mean()
    valid_loss = tf.metrics.Mean()

    for epoch in range(epoch_bias + 1, epoch_bias + epochs + 1):
        
        start_time = time.time()

        train_loss.reset_states()
        valid_loss.reset_states()
        
        for (x_target, y_target) in dataset_train:

            loss = train_step(x_target, y_target, model, optimizer, model_loss)
            train_loss.update_state(loss)

        for (x_target, y_target) in dataset_valid:

            loss = valid_step(x_target, y_target, model, model_loss)  
            valid_loss.update_state(loss)

        # if epoch > 10:
        
        #     optimizer.lr = 0.001 * tf.math.exp(0.1 * (10 - epoch))

        if epoch % saving_freq == 0:

            checkpoint.save(file_prefix = checkpoint_prefix)
                
        end_time = time.time()

        with open(f'{log_folder}/log_{model_name}.log','a') as fd:

            fd.write(f"{epoch},{train_loss.result().numpy()},{valid_loss.result().numpy()},{end_time - start_time}\n")

        print(f'Epoch {epoch:04d}/{(epoch_bias + epochs):04d}, loss: {train_loss.result().numpy()}, val_loss: {valid_loss.result().numpy()}, epoch time: {end_time - start_time}')


    return


@tf.function
def train_step(x_target, y_target, model, optimizer, model_loss):
    
    with tf.GradientTape() as model_tape:

        y_predic = model(x_target, training=True)

        loss = model_loss(y_target, y_predic)
    
    gradients_of_model = model_tape.gradient(loss, model.trainable_variables)
    
    optimizer.apply_gradients(zip(gradients_of_model, model.trainable_variables))
    
    return loss


@tf.function
def valid_step(x_target, y_target, model, model_loss):
    
    with tf.GradientTape() as model_tape:

        y_predic = model(x_target, training=False)

        loss = model_loss(y_target, y_predic)
    
    return loss