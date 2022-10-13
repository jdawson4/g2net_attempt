# Author: Jacob Dawson

import tensorflow as tf
from tensorflow import keras
import numpy as np
import h5py
#import os
import pandas as pd


data_files_path = "../g2net-detecting-continuous-gravitational-waves/"
train_path = data_files_path + "train/"
test_path = data_files_path + "test/"
train_dataset = tf.data.Dataset.list_files(train_path+"*.hdf5")

class GWaveDetector(keras.Model):
    def __init__(self, architecture, labels):
        super(GWaveDetector, self).__init__()
        self.architecture = architecture
        self.labels_df = pd.read_csv(data_files_path + 'train_labels.csv', header=1)
        self.loss_tracker = keras.metrics.Mean(name='loss')

    @property # no idea what this does
    def metrics(self):
        return [self.loss_tracker]
    
    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        
        # training here:
        with tf.GradientTape() as tape:
            with h5py.File(tf.as_string(data[0]), 'r') as f:
                gen_output = self.architecture(data, training=True)
        grads = tape.gradient(loss, self.architecture.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads,self.architecture.trainable_weights)
        )
        self.loss_tracker.update_state(loss)
        
        return {
            'loss': self.loss_tracker.result()
        }
