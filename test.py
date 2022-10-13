# Author: Jacob Dawson

import tensorflow as tf
#import tensorflow_datasets as tfds
import numpy as np
import h5py
import os

data_files_path = "../g2net-detecting-continuous-gravitational-waves/"
train_path = data_files_path + "train/"
test_path = data_files_path + "test/"
#print(os.listdir(train_path))

'''
for file in os.listdir(train_path):
    with h5py.File(train_path+file, 'r') as f:
        for file_key in f.keys():
            group = f[file_key]
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

    break
'''

#train_data = tfds.builder_from_directory(train_path).as_dataset()
#print(type(train_data.get_single_element()))

train_dataset = tf.data.Dataset.list_files(train_path+"*.hdf5")
print(train_dataset.cardinality())
print(train_dataset.take(1).get_single_element())
