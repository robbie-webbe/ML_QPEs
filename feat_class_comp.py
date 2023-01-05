#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 16:51:07 2022

@author: do19150
"""

import numpy as np
import pandas as pd
import sys


def class_deviation(obsid,dt):
    sim_qpe_feats = pd.read_csv('Features/qpe_feats_dt'+str(dt)+'.csv',header=None)
    sim_nqpe_feats = pd.read_csv('Features/no_qpe_feats_dt'+str(dt)+'.csv',header=None)
    real_feats = pd.read_csv('Features/realobs_test_data_dt'+str(dt)+'.csv',dtype='object')
    
    obs_feats = real_feats[real_feats['ObsID'] == obsid].values[0][1:-1].astype('float')
    print(obs_feats)
    
    qpe_feat_dev = []
    nqpe_feat_dev = []
    
    for i in range(14):
        
        qpe_feat_dev.append(abs((obs_feats[i] - np.average(sim_qpe_feats.iloc[:,i]))/np.std(sim_qpe_feats.iloc[:,i])))
        nqpe_feat_dev.append(abs((obs_feats[i] - np.average(sim_nqpe_feats.iloc[:,i]))/np.std(sim_nqpe_feats.iloc[:,i])))

    qpe_mean_dev = np.average(qpe_feat_dev)
    qpe_spread_dev = np.std(qpe_feat_dev)
    nqpe_mean_dev = np.average(np.ma.masked_invalid(nqpe_feat_dev))
    nqpe_spread_dev = np.std(np.ma.masked_invalid(nqpe_feat_dev))
    
    return qpe_mean_dev, qpe_spread_dev, nqpe_mean_dev, nqpe_spread_dev

print(class_deviation(sys.argv[1],sys.argv[2]))