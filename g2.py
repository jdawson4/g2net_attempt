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

@tf.function
def loss_fn(y_true, y_pred):
    # MSE for now
    tf.reduce_mean(tf.math.square(y_pred - y_true), axis=-1)

# can't read files from an @tf.function
@tf.py_function
def preprocess_hdf5(filename):
    with h5py.File(train_path + tf.compat.as_str_any(filename), 'r') as f:
        h1_key, l1_key, freq_key = f.keys() # get keys

        # get the three groups:
        h1_group = f[h1_key]
        l1_group = f[l1_key]
        freq_group = f[freq_key]
        try:
            for group_key in group.keys():
                print(np.array(group[group_key]))
                print(type(np.array(group[group_key])))
                print(np.array(group[group_key]).shape)
                print(np.array(group[group_key]).dtype)
                group2 = group[group_key]
                #print(f"---->{group2}")
                for group_key2 in group2.keys():
                    #print(f"--------->{group2[group_key2]}")
                    pass
        except AttributeError:
            pass

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
        for i in range(batch_size):
            hdf5 = preprocess_hdf5(data[i])
        
        # training here:
        with tf.GradientTape() as tape:
            gen_output = self.architecture(data, training=True)
            loss = loss_fn()
        grads = tape.gradient(loss, self.architecture.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads,self.architecture.trainable_weights)
        )
        self.loss_tracker.update_state(loss)
        
        return {
            'loss': self.loss_tracker.result()
        }
