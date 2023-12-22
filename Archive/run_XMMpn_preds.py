#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 14:08:07 2022

@author: do19150
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from astropy.io import fits
sys.path.append(os.getcwd()[:-6]+'Analysis_Funcs/General/')
from fitsloader import XMMtolc
from lcfeaturegen import lcfeat


#load the saved 14 feature dt250 and dt1000 models
dt50_model = tf.keras.models.load_model('saved_models/14_feats/feature_set0_dt50')
dt250_model = tf.keras.models.load_model('saved_models/14_feats/feature_set0_dt250')
dt1000_model = tf.keras.models.load_model('saved_models/14_feats/feature_set0_dt1000')
dt50_prob_model = tf.keras.Sequential([dt50_model, tf.keras.layers.Softmax()])
dt250_prob_model = tf.keras.Sequential([dt250_model, tf.keras.layers.Softmax()])
dt1000_prob_model = tf.keras.Sequential([dt1000_model, tf.keras.layers.Softmax()])

files = os.listdir('Obs/'+str(sys.argv[1])+'/pps/')

filename = []
for file in files:
    if file.startswith('P'):
        filename.append(file)
lc = XMMtolc('Obs/'+str(sys.argv[1])+'/pps/'+filename[0])

lc_50 = lc.rebin(50)
lc_250 = lc.rebin(250)
lc_1000 = lc.rebin(1000)
                                
                
feats_50 = np.asarray([list(lcfeat([lc_50.time,lc_50.countrate],qpe='?'))])
feats_250 = np.asarray([list(lcfeat([lc_250.time,lc_250.countrate],qpe='?'))])
feats_1000 = np.asarray([list(lcfeat([lc_1000.time,lc_1000.countrate],qpe='?'))])
                
pred_50 = dt50_prob_model.predict(feats_50)[0]
pred_250 = dt250_prob_model.predict(feats_250)[0]
pred_1000 = dt1000_prob_model.predict(feats_1000)[0] 

print(pred_50,pred_250,pred_1000)




