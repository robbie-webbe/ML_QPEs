#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:15:37 2024

@author: do19150
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from stingray import Lightcurve
from lcfeaturegen import lcfeat

#load the saved, trained, models
dt50_model = tf.keras.models.load_model('saved_models/14_feats/feature_set0_dt50')
dt250_model = tf.keras.models.load_model('saved_models/14_feats/feature_set0_dt250')
dt1k_model = tf.keras.models.load_model('saved_models/14_feats/feature_set0_dt1000')
dt50_prob_model = tf.keras.Sequential([dt50_model, tf.keras.layers.Softmax()])
dt250_prob_model = tf.keras.Sequential([dt250_model, tf.keras.layers.Softmax()])
dt1k_prob_model = tf.keras.Sequential([dt1k_model, tf.keras.layers.Softmax()])

#create a list of observations for which there are lightcurves
obslist = []
dir_list = sorted(os.listdir('EXOD/'))
for item in dir_list:
    if item.startswith('0'):
        obslist.append(item)
        
#create lists for all detections to add obsid, EXOD det id, and predictions
obsids = []
EXOD_id = []
avg_rate = []
preds50 = []
preds250 = []
preds1k = []

#iterate over every observation
for obs in obslist:
    print(obs) #for debugging
    #load the lightcurves to an array
    infile = 'EXOD/' + obs + '/lcs.csv'
    lcs = np.loadtxt(infile, delimiter=',',skiprows=1)
    
    if lcs.size == 0:
        continue
    
    #pick out the number of lightcurves for the observation
    if len(lcs.shape) == 1:
        #if only one lightcurve
        no_lcs = 1
        #pick out the length of the lightcurves
        lc_len = lcs.shape[0]
        
    else:
        no_lcs = len(lcs[0])
        #pick out the length of the lightcurves
        lc_len = len(lcs)
        
    #create an output time array
    times = np.arange(0,lc_len*50,50)
    
    #iterate over the detections in each observation
    for i in range(no_lcs):
        #print(i) #for debugging
        #update the output arrays for obslist and detection no.
        obsids.append(obs)
        EXOD_id.append(i)
        
        if no_lcs == 1:
            lc_raw = lcs[:]
        else:
            lc_raw = lcs[:,i]
        #update the average rate output
        avg_rate.append(np.average(lc_raw/50))
        
        #create a lightcurve object for the detection
        lc50 = Lightcurve(times, lc_raw)
        lc250 = lc50.rebin(250)
        lc1k = lc250.rebin(1000)
        
        #pick out the features for the detection
        try:
            feats_50 = np.asarray([list(lcfeat([lc50.time,lc50.countrate],qpe='?'))])
            feats_250 = np.asarray([list(lcfeat([lc250.time,lc250.countrate],qpe='?'))])
            feats_1k = np.asarray([list(lcfeat([lc1k.time,lc1k.countrate],qpe='?'))])
        except:
            preds50.append(0.0)
            preds250.append(0.0)
            preds1k.append(0.0)
        else:
            feats_50 = np.asarray([list(lcfeat([lc50.time,lc50.countrate],qpe='?'))])
            feats_250 = np.asarray([list(lcfeat([lc250.time,lc250.countrate],qpe='?'))])
            feats_1k = np.asarray([list(lcfeat([lc1k.time,lc1k.countrate],qpe='?'))])
            #predict the probability of there being a QPE in the detection
            pred_50 = dt50_prob_model.predict(feats_50,verbose=0)[0][1]
            pred_250 = dt250_prob_model.predict(feats_250,verbose=0)[0][1]
            pred_1k = dt1k_prob_model.predict(feats_1k,verbose=0)[0][1]
            #update the prediction output arrays
            preds50.append(pred_50)
            preds250.append(pred_250)
            preds1k.append(pred_1k)

        
#create an output dataframe        
out_df = pd.DataFrame() 
out_df['OBSID'] = obsids
out_df['EXOD_ID'] = EXOD_id
out_df['RATE'] = avg_rate
out_df['PRED50'] = preds50
out_df['PRED250'] = preds250
out_df['PRED1K'] = preds1k        

out_df.to_csv('EXOD/EXOD_preds.csv',index=False)
        
    
        
        