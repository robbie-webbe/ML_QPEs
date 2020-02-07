#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:51:59 2020

@author: do19150
"""

import numpy as np
import pandas as pd
import tensorflow as tf

csv_file = input('Enter location for training data set: ')
df = pd.read_csv(csv_file, names=['label','s1','s2','s3','s4','s5','s6','s7'],index_col=False)

labels = df.pop('label')

dataset = tf.data.Dataset.from_tensor_slices((df.values, labels.values))

train_dataset = dataset.shuffle(len(df)).batch(1)

def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
  return model

model = get_compiled_model()
model.fit(train_dataset, epochs=15)